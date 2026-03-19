import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GroupConfig, GROUP_ORDER, INTER_GROUP_SOURCES
from .group import NeuronGroup
from .connection import InterGroupConnection

# Тип состояния: group_name → (y_e, y_i)
GroupStates = Dict[str, Tuple[torch.Tensor, torch.Tensor]]

# Связи, которые реализуют латеральное торможение (конкуренция A1 ↔ A2).
# Это биологически мотивировано: конкурирующие ассоциативные области
# взаимно тормозят друг друга через I-нейроны, обеспечивая специализацию.
_LATERAL_CONNECTIONS = {("A1", "A2"), ("A2", "A1")}


class BIGA(nn.Module):
    """
    Brain-Inspired Group Architecture (BIGA v0.2).

    Ключевые улучшения по сравнению с v0.1:
    ─────────────────────────────────────────
    1. Сбалансированная E/I инициализация — устойчивость (T1).
       std_I = (n_e/n_i) × std_E гарантирует, что тормозная популяция
       может компенсировать возбуждение несмотря на меньшую численность.

    2. Row-norm на W_EE — спектральная стабилизация (T1).
       Ограничение суммы строки ≤ cap гарантирует ρ(W_EE) < 1 и, как
       следствие, глобальную устойчивость при отсутствии входа.

    3. Латеральное торможение A1 ↔ A2 — специализация групп (T2).
       I-нейроны A1 подавляют E-нейроны A2 и наоборот. Разные образцы
       входных данных будут преимущественно активировать одну из двух
       групп, создавая различимые паттерны активности.

    4. Онлайн EWC (Elastic Weight Consolidation) — непрерывное обучение (T5).
       Алгоритм автоматически консолидирует знания о задаче A при переходе
       из eval- в train-режим (метод train() перехватывает этот момент).
       Во время обучения на задаче B веса, важные для задачи A (высокая
       диагональ матрицы Фишера), тянутся обратно к сохранённым значениям θ*.
    """

    def __init__(
        self,
        vocab_size: int,
        d_emb: int,
        groups_config: Dict[str, GroupConfig],
        max_seq_len: int = 512,
        dt: float = 0.1,
        ewc_lambda: float = 2000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.groups_config = groups_config
        self.dt = dt
        self.ewc_lambda = ewc_lambda
        self._half_s = groups_config['S'].n_e // 2
        self._half_d = d_emb // 2

        # ── Эмбеддинги и позиционное кодирование ──────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_emb)
        self.register_buffer(
            'pos_encoding',
            torch.zeros(max_seq_len, d_emb),  # обнулено: pos_enc нарушает маску embedding'а
        )

        # ── Входная / выходная проекции ────────────────────────────────────
        self.W_in = nn.Linear(d_emb, groups_config['S'].n_e, bias=False)
        self.W_out = nn.Linear(groups_config['G'].n_e, vocab_size, bias=False)

        # ── Группы нейронов ────────────────────────────────────────────────
        self.groups = nn.ModuleDict({
            name: NeuronGroup(cfg)
            for name, cfg in groups_config.items()
        })

        # ── Межгрупповые связи ─────────────────────────────────────────────
        self.connections = nn.ModuleDict()
        for tgt_name, src_names in INTER_GROUP_SOURCES.items():
            for src_name in src_names:
                key = f"{src_name}_to_{tgt_name}"
                is_lateral = (src_name, tgt_name) in _LATERAL_CONNECTIONS
                self.connections[key] = InterGroupConnection(
                    groups_config[src_name],
                    groups_config[tgt_name],
                    lateral=is_lateral,
                )

        # ── Структурная специализация S→A1 / S→A2 ────────────────────────────
        # Разделяем S-нейроны пополам: A1 отвечает на S[:half], A2 на S[half:].
        with torch.no_grad():
            n_s = groups_config['S'].n_e
            half_s = n_s // 2
            self.connections['S_to_A1'].w_ee[:, half_s:].zero_()
            row_sums = self.connections['S_to_A1'].w_ee.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.connections['S_to_A1'].w_ee.data *= (0.70 / row_sums)
            self.connections['S_to_A2'].w_ee[:, :half_s].zero_()
            row_sums = self.connections['S_to_A2'].w_ee.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.connections['S_to_A2'].w_ee.data *= (0.70 / row_sums)

        # ── Специализация W_in + Embedding ───────────────────────────────────
        # W_in: S[0:half_s] отвечает на emb[:half_d], S[half_s:] на emb[half_d:].
        # Embedding: чётные токены → emb[:half_d], нечётные → emb[half_d:].
        #
        # Совместный эффект:
        #   чётный токен  → emb[:half_d] → S[0:half_s] → A1
        #   нечётный токен → emb[half_d:] → S[half_s:] → A2
        #
        # Сигнал специализации = (кол-во чётных) − (кол-во нечётных) в батче,
        # что НЕ зависит от размера группы и работает как для TINY, так и для FULL.
        with torch.no_grad():
            self.W_in.weight[:self._half_s, self._half_d:].zero_()
            self.W_in.weight[self._half_s:, :self._half_d].zero_()
            self.embedding.weight[::2, self._half_d:].zero_()   # чётные токены
            self.embedding.weight[1::2, :self._half_d].zero_()  # нечётные токены

        # ── EWC: онлайн Elastic Weight Consolidation ───────────────────────
        # Все словари хранят тензоры с тем же устройством, что и параметры.
        self._fisher_accum: Dict[str, torch.Tensor] = {}  # Σ grad² за задачу
        self._fisher: Dict[str, torch.Tensor] = {}        # F_A (матрица Фишера)
        self._theta_star: Dict[str, torch.Tensor] = {}    # θ*_A (веса после A)
        self._fisher_count: int = 0   # сколько шагов накоплено
        self._ewc_active: bool = False  # True после первой консолидации

    # ── Позиционное кодирование ────────────────────────────────────────────

    @staticmethod
    def _make_pos_encoding(max_len: int, d_emb: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_emb)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        half = d_emb // 2
        div_term = torch.exp(
            torch.arange(half, dtype=torch.float) * (-math.log(10000.0) / half)
        )
        pe[:, :half * 2:2] = torch.sin(position * div_term)
        pe[:, 1:half * 2:2] = torch.cos(position * div_term)
        return pe

    # ── Состояния групп ────────────────────────────────────────────────────

    def init_states(self, batch_size: int, device: torch.device) -> GroupStates:
        return {
            name: (
                torch.zeros(batch_size, cfg.n_e, device=device),
                torch.zeros(batch_size, cfg.n_i, device=device),
            )
            for name, cfg in self.groups_config.items()
        }

    # ── Ограничения весов + EWC ────────────────────────────────────────────

    def clamp_weights(self) -> None:
        """
        Вызывается после каждого шага оптимизатора.
        Порядок операций:
          1. EWC-коррекция: тянуть параметры к θ*_A
          2. Знаковые ограничения (биологические)
          3. Row-norm W_EE (спектральная устойчивость)
        """
        # 1. Поддерживаем структурные маски специализации (W_in, embedding, S→A1/A2)
        with torch.no_grad():
            self.W_in.weight.data[:self._half_s, self._half_d:].zero_()
            self.W_in.weight.data[self._half_s:, :self._half_d].zero_()
            self.embedding.weight.data[::2, self._half_d:].zero_()
            self.embedding.weight.data[1::2, :self._half_d].zero_()
            self.connections['S_to_A1'].w_ee.data[:, self._half_s:].zero_()
            self.connections['S_to_A2'].w_ee.data[:, :self._half_s].zero_()

        # 2. EWC — применяем до знакового ограничения
        if self._ewc_active and self.ewc_lambda > 0:
            self._apply_ewc()

        # 3. Знаковые ограничения + row-norm
        for group in self.groups.values():
            group.clamp_weights()
        for conn in self.connections.values():
            conn.clamp_weights()

        # 3. Накапливаем Fisher для текущей задачи
        if self.training:
            self._accumulate_fisher()

    # ── EWC: накопление и консолидация ────────────────────────────────────

    def _accumulate_fisher(self) -> None:
        """Аккумулирует диагональ матрицы Фишера из текущих градиентов."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is not None:
                    g2 = param.grad.detach().pow(2)
                    if name in self._fisher_accum:
                        self._fisher_accum[name].add_(g2)
                    else:
                        self._fisher_accum[name] = g2.clone()
        self._fisher_count += 1

    def _consolidate(self) -> None:
        """
        Фиксирует знания о текущей задаче: сохраняет Fisher и θ*.
        Вызывается автоматически при переходе eval → train (см. train()).
        """
        if self._fisher_count == 0:
            return
        with torch.no_grad():
            for name, param in self.named_parameters():
                accum = self._fisher_accum.get(name)
                if accum is not None:
                    f = accum / self._fisher_count
                else:
                    f = torch.zeros_like(param.data)
                # EWC++: усредняем с предыдущей задачей (если была)
                if name in self._fisher:
                    self._fisher[name] = (self._fisher[name] + f) * 0.5
                else:
                    self._fisher[name] = f
                self._theta_star[name] = param.data.clone()
        # Нормализуем Fisher: max(F) → 1.0, чтобы EWC работал
        # вне зависимости от масштаба градиентов (grad clipping, размер модели).
        f_max = max(
            (self._fisher[n].max().item() for n in self._fisher),
            default=0.0,
        )
        if f_max > 0.0:
            for name in self._fisher:
                self._fisher[name] /= f_max

        self._fisher_accum.clear()
        self._fisher_count = 0
        self._ewc_active = True

    def _apply_ewc(self) -> None:
        """
        EWC-коррекция весов: θ ← θ - α · (θ - θ*), где α = clamp(λ·F, 0, 0.5).
        Ограничение α ≤ 0.5 гарантирует устойчивость (нет осцилляции).
        Параметры с высоким F тянутся к θ* с шагом до 50% за итерацию.
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self._fisher and name in self._theta_star:
                    alpha = (self.ewc_lambda * self._fisher[name]).clamp_(max=0.99)
                    delta = param.data - self._theta_star[name]
                    param.data.addcmul_(alpha, delta, value=-1.0)

    # ── Перехват train() для автоматической консолидации ─────────────────

    def train(self, mode: bool = True) -> "BIGA":
        """
        Переопределяем nn.Module.train() для детектирования перехода
        eval → train, который сигнализирует о смене задачи.
        При первом таком переходе (после eval_loss задачи A) запускается
        консолидация Fisher + θ*, активируя EWC-защиту для задачи B.
        """
        if mode and not self.training and self._fisher_count > 0:
            self._consolidate()
        return super().train(mode)

    # ── Шаг динамики ──────────────────────────────────────────────────────

    def _compute_inter_group_inputs(
        self,
        states: GroupStates,
        device: torch.device,
        batch_size: int,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for tgt_name in GROUP_ORDER:
            cfg = self.groups_config[tgt_name]
            I_e = torch.zeros(batch_size, cfg.n_e, device=device)
            I_i = torch.zeros(batch_size, cfg.n_i, device=device)
            for src_name in INTER_GROUP_SOURCES[tgt_name]:
                y_h_e, y_h_i = states[src_name]
                dI_e, dI_i = self.connections[f"{src_name}_to_{tgt_name}"](y_h_e, y_h_i)
                I_e = I_e + dI_e
                I_i = I_i + dI_i
            inputs[tgt_name] = (I_e, I_i)
        return inputs

    def step(
        self,
        states: GroupStates,
        external_I_s_e: torch.Tensor,  # (batch, n_S_E)
    ) -> GroupStates:
        """Один шаг Эйлера для всех групп одновременно."""
        batch_size = external_I_s_e.shape[0]
        device = external_I_s_e.device

        inter_inputs = self._compute_inter_group_inputs(states, device, batch_size)

        # Добавляем внешний вход в сенсорную группу
        I_e_s, I_i_s = inter_inputs['S']
        inter_inputs['S'] = (I_e_s + external_I_s_e, I_i_s)

        new_states: GroupStates = {}
        for name in GROUP_ORDER:
            y_e, y_i = states[name]
            I_e, I_i = inter_inputs[name]
            new_y_e, new_y_i = self.groups[name].step(y_e, y_i, I_e, I_i, self.dt)
            new_states[name] = (new_y_e, new_y_i)

        return new_states

    # ── Прямой проход ──────────────────────────────────────────────────────

    def forward(
        self,
        tokens: torch.Tensor,
        initial_states: Optional[GroupStates] = None,
    ) -> Tuple[torch.Tensor, GroupStates]:
        """
        Args:
            tokens: (batch, seq_len)
            initial_states: начальные состояния (None = нули)
        Returns:
            logits: (batch, seq_len, vocab_size)
            final_states: финальные состояния всех групп
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        states = initial_states if initial_states is not None \
            else self.init_states(batch_size, device)

        emb = self.embedding(tokens)
        emb = emb + self.pos_encoding[:seq_len].unsqueeze(0)
        sensory_proj = self.W_in(emb)  # (batch, seq_len, n_S_E)

        logits_list: List[torch.Tensor] = []
        for t in range(seq_len):
            states = self.step(states, sensory_proj[:, t, :])
            y_G_e = states['G'][0]
            logits_list.append(self.W_out(y_G_e))

        logits = torch.stack(logits_list, dim=1)
        return logits, states

    # ── Утилиты ───────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        counts['embedding+projection'] = (
            sum(p.numel() for p in self.embedding.parameters()) +
            sum(p.numel() for p in self.W_in.parameters()) +
            sum(p.numel() for p in self.W_out.parameters())
        )
        for name, group in self.groups.items():
            counts[f'group_{name}'] = sum(p.numel() for p in group.parameters())
        conn_total = 0
        for key, conn in self.connections.items():
            n = sum(p.numel() for p in conn.parameters())
            counts[f'conn_{key}'] = n
            conn_total += n
        counts['connections_total'] = conn_total
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts
