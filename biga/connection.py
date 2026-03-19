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

    def clamp_weights(self) -> None:
        """Знаковые ограничения + row-norm на w_ee для спектральной устойчивости."""
        with torch.no_grad():
            self.w_ee.clamp_(min=0.0)
            self.w_ei.clamp_(max=0.0)
            self.w_ie.clamp_(min=0.0)
            self.w_ii.clamp_(max=0.0)

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
