"""
BIGA — Brain-Inspired Group Architecture (v0.1)
Демо: обучение на синтетических последовательностях токенов.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from biga import BIGA, GROUPS_TINY, GROUPS_SMALL, GROUPS_FULL


# ─── Конфигурация ────────────────────────────────────────────────────────────

VOCAB_SIZE    = 256      # байт-уровневая токенизация
D_EMB         = 64       # размерность эмбеддинга
BATCH_SIZE    = 8
SEQ_LEN       = 32
LEARNING_RATE = 1e-3
NUM_STEPS     = 200
LOG_EVERY     = 20

# Выберите конфигурацию: GROUPS_TINY / GROUPS_SMALL / GROUPS_FULL
GROUPS_CONFIG = GROUPS_TINY


# ─── Утилиты ─────────────────────────────────────────────────────────────────

def make_batch(vocab_size: int, batch_size: int, seq_len: int, device: torch.device):
    """Синтетические данные: случайные последовательности токенов."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def print_model_info(model: BIGA) -> None:
    counts = model.count_parameters()
    print("\n┌─ BIGA Model ─────────────────────────────────────────────────┐")
    for name, cfg in model.groups_config.items():
        grp_params = counts[f'group_{name}']
        print(f"│  Group {name:2s}: E={cfg.n_e:5d}  I={cfg.n_i:5d}"
              f"  τ_E={cfg.tau_e:5.1f}  τ_I={cfg.tau_i:5.1f}"
              f"  params={grp_params:>10,}")
    print(f"│")
    print(f"│  Connections: {counts['connections_total']:>12,} params")
    print(f"│  Emb+Proj:    {counts['embedding+projection']:>12,} params")
    print(f"│  ─────────────────────────────────────────────────────────────")
    print(f"│  TOTAL:       {counts['total']:>12,} params")
    print("└──────────────────────────────────────────────────────────────┘\n")


# ─── Обучение ─────────────────────────────────────────────────────────────────

def train_step(
    model: BIGA,
    tokens: torch.Tensor,
    optimizer: optim.Optimizer,
) -> float:
    """Один шаг обучения: предсказание следующего токена."""
    optimizer.zero_grad()

    logits, _ = model(tokens)  # (batch, seq_len, vocab_size)

    # Предсказываем tokens[t+1] по logits[t]
    logits_shifted = logits[:, :-1, :].contiguous().view(-1, model.vocab_size)
    targets = tokens[:, 1:].contiguous().view(-1)

    loss = nn.functional.cross_entropy(logits_shifted, targets)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Биологические ограничения на знаки весов (проекция после градиентного шага)
    model.clamp_weights()

    return loss.item()


# ─── Генерация ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: BIGA,
    prompt: list[int],
    max_new_tokens: int = 20,
    device: torch.device = torch.device('cpu'),
) -> list[int]:
    """Авторегрессивная генерация токенов."""
    model.eval()
    tokens = torch.tensor(prompt, device=device).unsqueeze(0)  # (1, len)
    states = model.init_states(1, device)

    generated = list(prompt)

    for _ in range(max_new_tokens):
        logits, states = model(tokens, initial_states=states)
        next_token = logits[0, -1, :].argmax().item()
        generated.append(next_token)
        tokens = torch.tensor([[next_token]], device=device)

    return generated


# ─── Главная функция ──────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    model = BIGA(
        vocab_size=VOCAB_SIZE,
        d_emb=D_EMB,
        groups_config=GROUPS_CONFIG,
        max_seq_len=SEQ_LEN + 16,
        dt=0.1,
    ).to(device)

    print_model_info(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Обучение BIGA | batch={BATCH_SIZE} seq_len={SEQ_LEN} steps={NUM_STEPS}")
    print(f"{'Шаг':>6}  {'Loss':>8}  {'Перплексия':>12}")
    print("─" * 35)

    model.train()
    for step in range(1, NUM_STEPS + 1):
        tokens = make_batch(VOCAB_SIZE, BATCH_SIZE, SEQ_LEN, device)
        loss = train_step(model, tokens, optimizer)

        if step % LOG_EVERY == 0 or step == 1:
            ppl = torch.exp(torch.tensor(loss)).item()
            print(f"{step:>6}  {loss:>8.4f}  {ppl:>12.2f}")

    print("\nОбучение завершено!")

    # Демо генерации
    print("\nДемо генерации (greedy) по промпту [65, 66, 67]:")
    result = generate(model, prompt=[65, 66, 67], max_new_tokens=10, device=device)
    print(f"  Токены: {result}")
    try:
        print(f"  Текст:  {''.join(chr(t) if 32 <= t < 127 else '?' for t in result)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
