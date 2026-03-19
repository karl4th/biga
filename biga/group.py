import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GroupConfig

# Максимально допустимая сумма строки W_EE.
# Гарантирует ρ(W_EE) ≤ ROW_SUM_CAP < 1 (достаточное условие устойчивости
# для неотрицательных матриц по теореме Перрона–Фробениуса).
_ROW_SUM_CAP = 0.30

# Максимальное количество входящих связей на нейрон внутри группы.
# Биологически правдоподобная разреженность: каждый нейрон соединён
# максимум с MAX_INTRA_CONNECTIONS другими нейронами группы.
_MAX_INTRA_CONNECTIONS = 100


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

        # Разреженность: максимум MAX_INTRA_CONNECTIONS входящих связей на нейрон
        self._create_sparse_masks(n_e, n_i)

    def _create_sparse_masks(self, n_e: int, n_i: int) -> None:
        """Создаёт бинарные маски для ограничения числа связей внутри группы."""
        with torch.no_grad():
            # Маска для W_EE: каждый E-нейрон получает ≤ MAX_CONNECTIONS входов от E
            max_conn_ee = min(_MAX_INTRA_CONNECTIONS, n_e)
            mask_ee = torch.zeros(n_e, n_e, dtype=torch.float32)
            for i in range(n_e):
                indices = torch.randperm(n_e)[:max_conn_ee]
                mask_ee[i, indices] = 1.0
            self.register_buffer('_sparse_mask_ee', mask_ee)

            # Маска для W_IE: каждый I-нейрон получает ≤ MAX_CONNECTIONS входов от E
            max_conn_ie = min(_MAX_INTRA_CONNECTIONS, n_e)
            mask_ie = torch.zeros(n_i, n_e, dtype=torch.float32)
            for i in range(n_i):
                indices = torch.randperm(n_e)[:max_conn_ie]
                mask_ie[i, indices] = 1.0
            self.register_buffer('_sparse_mask_ie', mask_ie)

            # Маска для W_EI: каждый E-нейрон получает ≤ MAX_CONNECTIONS входов от I
            max_conn_ei = min(_MAX_INTRA_CONNECTIONS, n_i)
            mask_ei = torch.zeros(n_e, n_i, dtype=torch.float32)
            for i in range(n_e):
                indices = torch.randperm(n_i)[:max_conn_ei]
                mask_ei[i, indices] = 1.0
            self.register_buffer('_sparse_mask_ei', mask_ei)

            # Маска для W_II: каждый I-нейрон получает ≤ MAX_CONNECTIONS входов от I
            max_conn_ii = min(_MAX_INTRA_CONNECTIONS, n_i)
            mask_ii = torch.zeros(n_i, n_i, dtype=torch.float32)
            for i in range(n_i):
                indices = torch.randperm(n_i)[:max_conn_ii]
                mask_ii[i, indices] = 1.0
            self.register_buffer('_sparse_mask_ii', mask_ii)

            # Применяем маски сразу при инициализации
            self.W_EE.data *= mask_ee
            self.W_EI.data *= mask_ei
            self.W_IE.data *= mask_ie
            self.W_II.data *= mask_ii

    def clamp_weights(self) -> None:
        """
        Применяет три ограничения:
        1. Знаковые (биологические): E-связи ≥ 0, I-связи ≤ 0.
        2. Разреженность: максимум MAX_CONNECTIONS входящих связей на нейрон.
        3. Устойчивость (спектральная): строчные суммы W_EE ≤ ROW_SUM_CAP.
           При выполнении этого условия ρ(W_EE) ≤ ROW_SUM_CAP < 1, что
           обеспечивает глобальную устойчивость внутригрупповой динамики.
        """
        with torch.no_grad():
            # Знаковые ограничения
            self.W_EE.clamp_(min=0.0)
            self.W_EI.clamp_(max=0.0)
            self.W_IE.clamp_(min=0.0)
            self.W_II.clamp_(max=0.0)

            # Разреженность: применяем маски
            if hasattr(self, '_sparse_mask_ee'):
                self.W_EE.data *= self._sparse_mask_ee
                self.W_EI.data *= self._sparse_mask_ei
                self.W_IE.data *= self._sparse_mask_ie
                self.W_II.data *= self._sparse_mask_ii

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
