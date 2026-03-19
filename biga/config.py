from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GroupConfig:
    name: str
    n_e: int          # количество возбуждающих нейронов
    n_i: int          # количество тормозных нейронов
    tau_e: float      # постоянная времени для E
    tau_i: float      # постоянная времени для I


# Полная конфигурация согласно спецификации
GROUPS_FULL: Dict[str, GroupConfig] = {
    'S':  GroupConfig('S',   1000,  200,  1.0,  0.8),
    'A1': GroupConfig('A1',  2000,  400,  1.0,  0.8),
    'A2': GroupConfig('A2',  2000,  400,  1.0,  0.8),
    'M':  GroupConfig('M',   5000, 1000, 10.0,  8.0),
    'G':  GroupConfig('G',   1000,  200,  1.0,  0.8),
}

# Уменьшенная конфигурация для тестирования
GROUPS_TINY: Dict[str, GroupConfig] = {
    'S':  GroupConfig('S',    64,   13,  1.0,  0.8),
    'A1': GroupConfig('A1',  128,   26,  1.0,  0.8),
    'A2': GroupConfig('A2',  128,   26,  1.0,  0.8),
    'M':  GroupConfig('M',   256,   52, 10.0,  8.0),
    'G':  GroupConfig('G',    64,   13,  1.0,  0.8),
}

# Средняя конфигурация (10% от полной)
GROUPS_SMALL: Dict[str, GroupConfig] = {
    'S':  GroupConfig('S',   100,   20,  1.0,  0.8),
    'A1': GroupConfig('A1',  200,   40,  1.0,  0.8),
    'A2': GroupConfig('A2',  200,   40,  1.0,  0.8),
    'M':  GroupConfig('M',   500,  100, 10.0,  8.0),
    'G':  GroupConfig('G',   100,   20,  1.0,  0.8),
}

# Порядок обработки групп (прямой проход)
GROUP_ORDER: List[str] = ['S', 'A1', 'A2', 'M', 'G']

# Межгрупповые связи: группа → список источников
# Самосвязи (рекуррентность) реализованы через внутригрупповые матрицы W_EE, W_EI, W_IE, W_II
INTER_GROUP_SOURCES: Dict[str, List[str]] = {
    'S':  [],                  # только внешний вход
    'A1': ['S', 'A2', 'M'],    # S, A2, M → A1 (рекуррентность A1 через W_g^{**})
    'A2': ['S', 'A1', 'M'],    # S, A1, M → A2
    'M':  ['A1', 'A2'],        # A1, A2 → M
    'G':  ['A1', 'A2', 'M'],   # A1, A2, M → G (рекуррентность G через W_g^{**})
}
