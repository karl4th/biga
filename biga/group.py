import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GroupConfig

# Максимально допустимая сумма строки W_EE.
# Гарантирует ρ(W_EE) ≤ ROW_SUM_CAP < 1 (достаточное условие устойчивости
# для неотрицательных матриц по теореме Перрона–Фробениуса).
_ROW_SUM_CAP = 0.30


class NeuronGroup(nn.Module):
    """
    Группа нейронов с возбуждающей (E) и тормозной (I) популяциями.

    Динамика:
        τ_E · dy_E/dt = -y_E + ReLU(W_EE·y_E + W_EI·y_I + b_E + I_E(t))
        τ_I · dy_I/dt = -y_I + ReLU(W_IE·y_E + W_II·y_I + b_I + I_I(t))

    Соглашение W^{target, source}:
        W_EE: E→E, (n_e, n_e), W ≥ 0
        W_EI: I→E, (n_e, n_i), W ≤ 0
        W_IE: E→I, (n_i, n_e), W ≥ 0
        W_II: I→I, (n_i, n_i), W ≤ 0

    Инициализация «сбалансированная сеть»:
        std_E = 0.5 / n_e          — гарантирует ρ(W_EE) ≤ 0.4 при инит.
        std_I = (n_e / n_i) * std_E — I-нейроны компенсируют ~5x меньшую
                                       численность тормозной популяции
    """

    def __init__(self, config: GroupConfig):
        super().__init__()
        self.name = config.name
        self.n_e = config.n_e
        self.n_i = config.n_i
        self.tau_e = config.tau_e
        self.tau_i = config.tau_i

        n_e, n_i = config.n_e, config.n_i

        # Стандартные отклонения: сбалансированная инициализация
        std_e = 0.5 / n_e           # возбуждающие связи
        std_i = (n_e / n_i) * std_e  # тормозные — пропорционально больше

        self.W_EE = nn.Parameter(torch.abs(torch.randn(n_e, n_e) * std_e))
        self.W_EI = nn.Parameter(-torch.abs(torch.randn(n_e, n_i) * std_i))
        self.W_IE = nn.Parameter(torch.abs(torch.randn(n_i, n_e) * std_e))
        self.W_II = nn.Parameter(-torch.abs(torch.randn(n_i, n_i) * std_i))

        self.b_E = nn.Parameter(torch.zeros(n_e))
        self.b_I = nn.Parameter(torch.zeros(n_i))

    def clamp_weights(self) -> None:
        """
        Применяет два ограничения:
        1. Знаковые (биологические): E-связи ≥ 0, I-связи ≤ 0.
        2. Устойчивость (спектральная): строчные суммы W_EE ≤ ROW_SUM_CAP.
           При выполнении этого условия ρ(W_EE) ≤ ROW_SUM_CAP < 1, что
           обеспечивает глобальную устойчивость внутригрупповой динамики.
        """
        with torch.no_grad():
            # Знаковые ограничения
            self.W_EE.clamp_(min=0.0)
            self.W_EI.clamp_(max=0.0)
            self.W_IE.clamp_(min=0.0)
            self.W_II.clamp_(max=0.0)

            # Спектральная стабилизация: нормируем строки W_EE
            row_sums = self.W_EE.sum(dim=1, keepdim=True)
            self.W_EE.data /= torch.clamp(row_sums / _ROW_SUM_CAP, min=1.0)

    def step(
        self,
        y_e: torch.Tensor,   # (batch, n_e)
        y_i: torch.Tensor,   # (batch, n_i)
        I_e: torch.Tensor,   # (batch, n_e) — межгрупповой вход в E
        I_i: torch.Tensor,   # (batch, n_i) — межгрупповой вход в I
        dt: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Один шаг метода Эйлера. Возвращает (y_e_new, y_i_new)."""
        net_e = y_e @ self.W_EE.T + y_i @ self.W_EI.T + self.b_E + I_e
        net_i = y_e @ self.W_IE.T + y_i @ self.W_II.T + self.b_I + I_i

        dy_e = (-y_e + F.relu(net_e)) / self.tau_e
        dy_i = (-y_i + F.relu(net_i)) / self.tau_i

        y_e_new = F.relu(y_e + dt * dy_e)
        y_i_new = F.relu(y_i + dt * dy_i)

        return y_e_new, y_i_new

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, n_e={self.n_e}, n_i={self.n_i}, "
            f"tau_e={self.tau_e}, tau_i={self.tau_i}"
        )
