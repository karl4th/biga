import math

import torch
import torch.nn as nn

from .config import GroupConfig

# Row-sum cap for excitatory inter-group weights (w_ee).
# Each target neuron can receive at most cap × mean_source_activity.
# For the A1→M→A1 feedback loop: loop_gain = cap² = 0.49 < 1 → stable.
# With cap=0.70 the signal chain S→A1→G delivers ~0.04 per neuron to G,
# making G output non-trivial (required for noise-robustness test T4).
_ROW_SUM_CAP = 0.70

# Maximum number of incoming connections per neuron.
# Biologically plausible sparse connectivity: each neuron connects to
# at most MAX_CONNECTIONS source neurons (random subset selected at init).
# This reduces memory and computation for large groups (GROUPS_FULL).
_MAX_CONNECTIONS = 100

# Lateral inhibition scale vs standard feedforward init.
# Both pathways scaled: w_ei (I_src→E_tgt) and w_ie (E_src→I_tgt).
# Scale 6 is strong enough for winner-take-all specialisation (T2) while
# keeping competition decisions stable under σ=0.1 input noise (T4).
_LATERAL_SCALE = 60.0


class InterGroupConnection(nn.Module):
    """
    Связи от исходной группы h к целевой группе g.

    Вычисляет:
        I_{h→g}^E = W_EE · y_h^E + W_EI · y_h^I   (вход в E-нейроны g)
        I_{h→g}^I = W_IE · y_h^E + W_II · y_h^I   (вход в I-нейроны g)

    lateral=True — латеральное торможение A1 ↔ A2 (winner-take-all конкуренция).
    Используются два биологических механизма:
      1. w_ei LARGE: I-нейроны источника напрямую тормозят E-нейроны цели
                     (src_I → tgt_E за 1 шаг от src_E).
      2. w_ie LARGE: E-нейроны источника возбуждают I-нейроны цели
                     (src_E → tgt_I → tgt_E за 2 шага, feedforward inhibition).
      3. w_ee ≈ 0:  нет прямого возбуждения между конкурирующими группами.
      4. w_ii ≈ 0:  нет I→I пути, который создаёт дис-ингибицию (разблокировку)
                     и ослабляет конкуренцию.
    """

    def __init__(self, src: GroupConfig, tgt: GroupConfig, lateral: bool = False):
        super().__init__()
        self._src_name = src.name
        self._tgt_name = tgt.name
        self._lateral = lateral

        n_h_e, n_h_i = src.n_e, src.n_i
        n_g_e, n_g_i = tgt.n_e, tgt.n_i

        # Стандартные отклонения.
        # E[row_sum w_ee] = n_h_e * std_ee * sqrt(2/π) ≈ ROW_SUM_CAP — насыщаем cap сразу,
        # чтобы незаученная модель уже имела реальный сигнал (нужно для T4).
        _sqrt2_over_pi = math.sqrt(2.0 / math.pi)  # ≈ 0.798
        std_ee = _ROW_SUM_CAP / (n_h_e * _sqrt2_over_pi)
        std_ei = (n_h_e / n_h_i) * std_ee   # E/I баланс (для латеральных связей)
        std_ie = std_ee
        std_ii = std_ei

        if lateral:
            # Латеральное торможение: только w_ei крупное.
            # w_ie ≈ 0: большой w_ie создаёт петлю самоподавления
            #   E_src → I_tgt → (lateral w_ei обратно) → E_src,
            #   которая симметрично давит оба конкурирующих паттерна
            #   и препятствует выбору победителя.
            # При w_ie≈0 только w_ei (I_src → E_tgt) обеспечивает
            #   асимметричное WTA-торможение без самоподавления.
            std_ee *= 0.02
            std_ie *= 0.02
            std_ei *= _LATERAL_SCALE
            std_ii *= 0.02
        else:
            # Прямые (feedforward) связи: только w_ee сильное.
            # w_ie и w_ei ≈ 0: если бы S→A1 w_ie было большим, S сильно
            # возбуждал бы A1_I, которые затем через латеральный w_ei(A1→A2)
            # подавляли бы A2 — сигнал конкуренции смешивался бы с сенсорным
            # входом, уничтожая WTA. Нулевые w_ie/w_ei/w_ii изолируют
            # I-нейроны группы от внешнего вмешательства.
            std_ie *= 0.02
            std_ei *= 0.02
            std_ii *= 0.02

        self.w_ee = nn.Parameter(torch.abs(torch.randn(n_g_e, n_h_e) * std_ee))
        self.w_ei = nn.Parameter(-torch.abs(torch.randn(n_g_e, n_h_i) * std_ei))
        self.w_ie = nn.Parameter(torch.abs(torch.randn(n_g_i, n_h_e) * std_ie))
        self.w_ii = nn.Parameter(-torch.abs(torch.randn(n_g_i, n_h_i) * std_ii))

        # Разреженность: максимум MAX_CONNECTIONS входящих связей на нейрон
        # Создаём маску для w_ee (n_g_e × n_h_e)
        self._create_sparse_masks(n_g_e, n_h_e, n_g_i, n_h_i)

    def _create_sparse_masks(
        self,
        n_g_e: int, n_h_e: int,
        n_g_i: int, n_h_i: int,
    ) -> None:
        """Создаёт бинарные маски для ограничения числа связей."""
        with torch.no_grad():
            # Маска для w_ee: каждый целевой E-нейрон получает ≤ MAX_CONNECTIONS входов
            max_conn_e = min(_MAX_CONNECTIONS, n_h_e)
            mask_ee = torch.zeros(n_g_e, n_h_e, dtype=torch.float32)
            for i in range(n_g_e):
                indices = torch.randperm(n_h_e)[:max_conn_e]
                mask_ee[i, indices] = 1.0
            self.register_buffer('_sparse_mask_ee', mask_ee)

            # Маска для w_ie: каждый целевой I-нейрон получает ≤ MAX_CONNECTIONS входов
            max_conn_i = min(_MAX_CONNECTIONS, n_h_e)
            mask_ie = torch.zeros(n_g_i, n_h_e, dtype=torch.float32)
            for i in range(n_g_i):
                indices = torch.randperm(n_h_e)[:max_conn_i]
                mask_ie[i, indices] = 1.0
            self.register_buffer('_sparse_mask_ie', mask_ie)

            # Маска для w_ei: каждый целевой E-нейрон получает ≤ MAX_CONNECTIONS входов от I
            max_conn_ei = min(_MAX_CONNECTIONS, n_h_i)
            mask_ei = torch.zeros(n_g_e, n_h_i, dtype=torch.float32)
            for i in range(n_g_e):
                indices = torch.randperm(n_h_i)[:max_conn_ei]
                mask_ei[i, indices] = 1.0
            self.register_buffer('_sparse_mask_ei', mask_ei)

            # Маска для w_ii: каждый целевой I-нейрон получает ≤ MAX_CONNECTIONS входов от I
            max_conn_ii = min(_MAX_CONNECTIONS, n_h_i)
            mask_ii = torch.zeros(n_g_i, n_h_i, dtype=torch.float32)
            for i in range(n_g_i):
                indices = torch.randperm(n_h_i)[:max_conn_ii]
                mask_ii[i, indices] = 1.0
            self.register_buffer('_sparse_mask_ii', mask_ii)

            # Применяем маски сразу при инициализации
            self.w_ee.data *= mask_ee
            self.w_ei.data *= mask_ei
            self.w_ie.data *= mask_ie
            self.w_ii.data *= mask_ii

    def clamp_weights(self) -> None:
        """Знаковые ограничения + row-norm на w_ee + разреженность (max 100 связей)."""
        with torch.no_grad():
            self.w_ee.clamp_(min=0.0)
            self.w_ei.clamp_(max=0.0)
            self.w_ie.clamp_(min=0.0)
            self.w_ii.clamp_(max=0.0)

            # Разреженность: максимум MAX_CONNECTIONS входящих связей на нейрон
            # Применяем маску разреженности при инициализации и после каждого шага
            if hasattr(self, '_sparse_mask_ee') and self._sparse_mask_ee is not None:
                self.w_ee.data *= self._sparse_mask_ee
                self.w_ei.data *= self._sparse_mask_ee
                self.w_ie.data *= self._sparse_mask_ie if hasattr(self, '_sparse_mask_ie') else 1.0
                self.w_ii.data *= self._sparse_mask_ii if hasattr(self, '_sparse_mask_ii') else 1.0

            row_sums = self.w_ee.sum(dim=1, keepdim=True)
            self.w_ee.data /= torch.clamp(row_sums / _ROW_SUM_CAP, min=1.0)

    def forward(
        self,
        y_h_e: torch.Tensor,  # (batch, n_h_E)
        y_h_i: torch.Tensor,  # (batch, n_h_I)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        I_g_e = y_h_e @ self.w_ee.T + y_h_i @ self.w_ei.T
        I_g_i = y_h_e @ self.w_ie.T + y_h_i @ self.w_ii.T
        return I_g_e, I_g_i

    def extra_repr(self) -> str:
        mode = "lateral" if self._lateral else "feedforward"
        return f"{self._src_name}→{self._tgt_name} [{mode}]"
