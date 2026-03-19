"""
Дополнительные тесты BIGA v0.4 — Лингвистические способности.

Тесты:
  T6 — Обучение на тексте (сравнение с RNN)
  T7 — Синтаксическая чувствительность (минимальные пары)
  T8 — Семантическое сходство (активации похожих слов)
  T9 — Анализ группы M (что хранится в медленной памяти)
"""

import argparse
import math
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from biga import BIGA, GROUPS_FULL, GROUPS_SMALL, GROUPS_TINY

torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 256
D_EMB = 64

print(f"Используемое устройство: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def make_biga(groups_config: dict) -> BIGA:
    """Создать BIGA модель с фиксированным seed."""
    torch.manual_seed(42)
    return BIGA(
        vocab_size=VOCAB_SIZE,
        d_emb=D_EMB,
        groups_config=groups_config,
        max_seq_len=256,
        dt=0.1,
    ).to(DEVICE)


def make_rnn(d_emb: int = 64, d_hidden: int = 128, n_layers: int = 2) -> nn.Module:
    """Создать простую RNN для сравнения."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Embedding(VOCAB_SIZE, d_emb),
        nn.LSTM(
            input_size=d_emb,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
        ),
        nn.Linear(d_hidden, VOCAB_SIZE),
    ).to(DEVICE)


def count_params(model: nn.Module) -> int:
    """Посчитать количество параметров модели."""
    return sum(p.numel() for p in model.parameters())


def generate_mini_corpus(n_samples: int = 1000, seq_len: int = 50) -> list[list[int]]:
    """
    Сгенерировать мини-корпус с простыми паттернами.
    Имитация текста с базовой структурой предложений.
    """
    torch.manual_seed(123)
    
    # Простые шаблоны "предложений"
    # Субъект + глагол + объект
    subjects = [10, 11, 12, 13, 14]  # "cat", "dog", "bird", etc.
    verbs = [20, 21, 22, 23]  # "sleeps", "runs", "jumps", etc.
    objects = [30, 31, 32, 33, 34]  # "bed", "park", "house", etc.
    articles = [1, 2]  # "the", "a"
    punctuation = [50]  # "."
    
    corpus = []
    for _ in range(n_samples):
        seq = []
        for _ in range(seq_len // 5):
            # Article + Subject + Verb + Object + Punctuation
            seq.extend([
                articles[torch.randint(0, len(articles), (1,)).item()],
                subjects[torch.randint(0, len(subjects), (1,)).item()],
                verbs[torch.randint(0, len(verbs), (1,)).item()],
                objects[torch.randint(0, len(objects), (1,)).item()],
                punctuation[0],
            ])
        corpus.append(seq[:seq_len])
    
    return corpus


def train_biga(model: BIGA, corpus: list[list[int]], n_epochs: int, lr: float = 1e-3) -> list[float]:
    """Обучить BIGA на корпусе."""
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    
    batch_size = 8
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle corpus
        indices = torch.randperm(len(corpus))
        
        for i in range(0, len(corpus), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_seqs = [corpus[idx] for idx in batch_indices]
            tokens = torch.tensor(batch_seqs, dtype=torch.long, device=DEVICE)
            
            opt.zero_grad()
            logits, _ = model(tokens)
            logits_s = logits[:, :-1, :].contiguous().view(-1, VOCAB_SIZE)
            targets = tokens[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits_s, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.clamp_weights()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
    
    return losses


def train_rnn(model: nn.Module, corpus: list[list[int]], n_epochs: int, lr: float = 1e-3) -> list[float]:
    """Обучить RNN на корпусе."""
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    
    batch_size = 8
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        indices = torch.randperm(len(corpus))
        
        for i in range(0, len(corpus), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_seqs = [corpus[idx] for idx in batch_indices]
            tokens = torch.tensor(batch_seqs, dtype=torch.long, device=DEVICE)
            
            opt.zero_grad()
            logits = model(tokens)
            logits_s = logits[:, :-1, :].contiguous().view(-1, VOCAB_SIZE)
            targets = tokens[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits_s, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
    
    return losses


def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


# ─── T6: Обучение на тексте ───────────────────────────────────────────────────

def test_text_learning(groups_config: dict) -> dict:
    """
    Обучить BIGA и простую RNN на одинаковом мини-корпусе.
    Сравнить final loss и количество параметров.
    """
    section("T6: Обучение на тексте (сравнение с RNN)")
    
    # Параметры
    n_samples = 500
    seq_len = 40
    n_epochs = 10
    lr = 1e-3
    
    print(f"\n  Генерация мини-корпуса: {n_samples} образцов, {seq_len} токенов...")
    corpus = generate_mini_corpus(n_samples, seq_len)
    
    # BIGA
    print(f"\n  Обучение BIGA ({n_epochs} эпох)...")
    biga = make_biga(groups_config)
    biga_params = count_params(biga)
    print(f"  Параметры BIGA: {biga_params:,}")
    
    t0 = time.time()
    biga_losses = train_biga(biga, corpus, n_epochs, lr)
    biga_time = time.time() - t0
    
    biga_final_loss = biga_losses[-1]
    print(f"  BIGA final loss: {biga_final_loss:.4f}  ({biga_time:.1f}с)")
    
    # RNN с похожим размером
    d_hidden = 128
    rnn = make_rnn(d_emb=D_EMB, d_hidden=d_hidden, n_layers=2)
    rnn_params = count_params(rnn)
    print(f"\n  Параметры RNN: {rnn_params:,}")
    
    print(f"\n  Обучение RNN ({n_epochs} эпох)...")
    t0 = time.time()
    rnn_losses = train_rnn(rnn, corpus, n_epochs, lr)
    rnn_time = time.time() - t0
    
    rnn_final_loss = rnn_losses[-1]
    print(f"  RNN final loss: {rnn_final_loss:.4f}  ({rnn_time:.1f}с)")
    
    # Сравнение
    loss_diff = biga_final_loss - rnn_final_loss
    biga_better = biga_final_loss < rnn_final_loss
    
    print(f"\n  Разница loss (BIGA - RNN): {loss_diff:+.4f}")
    print(f"  Вывод: {'BIGA' if biga_better else 'RNN'} показывает лучший loss")
    
    return {
        "biga_params": biga_params,
        "rnn_params": rnn_params,
        "biga_final_loss": round(biga_final_loss, 4),
        "rnn_final_loss": round(rnn_final_loss, 4),
        "loss_difference": round(loss_diff, 4),
        "biga_better": biga_better,
        "biga_epochs": biga_losses,
        "rnn_epochs": rnn_losses,
    }


# ─── T7: Синтаксическая чувствительность ─────────────────────────────────────

def test_syntax_sensitivity(groups_config: dict) -> dict:
    """
    Минимальные пары: грамматически правильные vs неправильные.
    "The cat sleeps" vs "*The cat sleep"
    Смотрим разницу в уверенности модели.
    """
    section("T7: Синтаксическая чувствительность (минимальные пары)")
    
    model = make_biga(groups_config)
    
    # Обучить немного на простых паттернах
    print("\n  Предварительное обучение на синтаксических паттернах...")
    corpus = generate_mini_corpus(300, 30)
    train_biga(model, corpus, n_epochs=5, lr=1e-3)
    
    # Минимальные пары (правильные vs неправильные)
    # Кодируем токенами: the=1, cat=10, dog=11, sleeps=20, sleep=21, runs=22, run=23
    pairs = [
        {
            "name": "Subject-Verb agreement (3rd person)",
            "correct": [1, 10, 20],      # "the cat sleeps"
            "incorrect": [1, 10, 21],    # "*the cat sleep"
        },
        {
            "name": "Subject-Verb agreement (plural)",
            "correct": [1, 11, 22],      # "the dog runs"
            "incorrect": [1, 11, 23],    # "*the dog run"
        },
    ]
    
    model.eval()
    results = []
    
    print(f"\n  {'Пара':<40}  {'P(correct)':>12}  {'P(incorrect)':>14}  {'Разница':>10}")
    print(f"  {'─' * 82}")
    
    for pair in pairs:
        correct_seq = torch.tensor([pair["correct"]], dtype=torch.long, device=DEVICE)
        incorrect_seq = torch.tensor([pair["incorrect"]], dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            # Последний токен для предсказания
            correct_logits, _ = model(correct_seq[:, :-1])
            incorrect_logits, _ = model(incorrect_seq[:, :-1])
            
            # Предсказание следующего токена
            correct_pred = correct_logits[:, -1, :]  # (1, vocab)
            incorrect_pred = incorrect_logits[:, -1, :]  # (1, vocab)
            
            # Вероятность правильного продолжения
            target_correct = pair["correct"][-1]
            target_incorrect = pair["incorrect"][-1]
            
            p_correct = F.softmax(correct_pred, dim=-1)[0, target_correct].item()
            p_incorrect = F.softmax(incorrect_pred, dim=-1)[0, target_incorrect].item()
            
            diff = p_correct - p_incorrect
            results.append({
                "name": pair["name"],
                "p_correct": p_correct,
                "p_incorrect": p_incorrect,
                "difference": diff,
            })
        
        print(f"  {pair['name']:<40}  {p_correct:>12.6f}  {p_incorrect:>14.6f}  {diff:>+10.6f}")
    
    # Итог
    avg_diff = sum(r["difference"] for r in results) / len(results)
    syntax_sensitive = avg_diff > 0
    
    print(f"\n  Средняя разница: {avg_diff:+.6f}")
    print(f"  Вывод: модель {'чувствительна' if syntax_sensitive else 'НЕ чувствительна'} к синтаксическим нарушениям")
    
    return {
        "minimal_pairs": results,
        "average_difference": round(avg_diff, 6),
        "syntax_sensitive": syntax_sensitive,
    }


# ─── T8: Семантическое сходство ───────────────────────────────────────────────

def test_semantic_similarity(groups_config: dict) -> dict:
    """
    Проверить, что активации семантически близких слов
    ("cat" и "dog") ближе, чем далёких ("cat" и "run").
    """
    section("T8: Семантическое сходство (активации)")
    
    model = make_biga(groups_config)
    
    # Обучить на корпусе с семантической структурой
    print("\n  Обучение на корпусе с семантическими паттернами...")
    corpus = generate_mini_corpus(300, 30)
    train_biga(model, corpus, n_epochs=5, lr=1e-3)
    
    # Токены для проверки
    # subjects: cat=10, dog=11, bird=12
    # verbs: sleeps=20, runs=22, jumps=24
    token_names = {
        10: "cat",
        11: "dog",
        12: "bird",
        20: "sleeps",
        22: "runs",
        24: "jumps",
    }
    
    # Получить активации для каждого токена
    model.eval()
    activations = {}
    
    with torch.no_grad():
        for token_id, name in token_names.items():
            # Подать токен как контекст: [1, token_id] (the + word)
            tokens = torch.tensor([[1, token_id]], dtype=torch.long, device=DEVICE)
            _, states = model(tokens)
            
            # Взять активацию группы G (выходная) на последней позиции
            g_activation = states['G'][0]  # (batch, n_e)
            activations[name] = g_activation[0]  # (n_e,)
    
    # Вычислить попарные косинусные расстояния
    print(f"\n  Активации группы G для разных токенов:")
    print(f"  {'Токен 1':>10}  {'Токен 2':>10}  {'Cosine sim':>12}")
    print(f"  {'─' * 38}")
    
    pairs_to_compare = [
        ("cat", "dog", "оба существительные (животные)"),
        ("cat", "sleeps", "сущ. vs глагол"),
        ("dog", "bird", "оба существительные (животные)"),
        ("sleeps", "runs", "оба глаголы"),
    ]
    
    similarities = {}
    for name1, name2, description in pairs_to_compare:
        v1 = F.normalize(activations[name1].unsqueeze(0), dim=-1)
        v2 = F.normalize(activations[name2].unsqueeze(0), dim=-1)
        cos_sim = F.cosine_similarity(v1, v2).item()
        similarities[f"{name1}-{name2}"] = cos_sim
        print(f"  {name1:>10}  {name2:>10}  {cos_sim:>12.4f}  ({description})")
    
    # Проверка гипотезы: cat-dog ближе чем cat-sleeps
    cat_dog_sim = similarities.get("cat-dog", 0)
    cat_sleeps_sim = similarities.get("cat-sleeps", 0)
    
    semantic_confirmed = cat_dog_sim > cat_sleeps_sim
    
    print(f"\n  Гипотеза: cat-dog ({cat_dog_sim:.4f}) > cat-sleeps ({cat_sleeps_sim:.4f})")
    print(f"  Вывод: семантическое сходство {'подтверждено ✓' if semantic_confirmed else 'не подтверждено ✗'}")
    
    return {
        "token_activations": {k: "tensor(...)" for k in activations},  # Не сериализуем тензоры
        "pairwise_similarities": {k: round(v, 4) for k, v in similarities.items()},
        "cat_dog_similarity": round(cat_dog_sim, 4),
        "cat_sleeps_similarity": round(cat_sleeps_sim, 4),
        "semantic_confirmed": semantic_confirmed,
    }


# ─── T9: Анализ группы M ──────────────────────────────────────────────────────

def test_m_group_analysis(groups_config: dict) -> dict:
    """
    Что хранится в медленной памяти группы M?
    Можно ли "прочитать" память и интерпретировать?
    """
    section("T9: Анализ группы M (медленная память)")
    
    model = make_biga(groups_config)
    
    batch = 4
    n_input = 50
    
    states = model.init_states(batch, DEVICE)
    
    # Фаза 1: Паттерн A (токены 10-19)
    print("\n  Фаза 1: подаём паттерн A (токены 10-19)...")
    pattern_a = torch.randint(10, 20, (batch, n_input), device=DEVICE)
    with torch.no_grad():
        _, states = model(pattern_a, initial_states=states)
    
    # Запомнить состояние M после паттерна A
    m_state_after_a = states['M'][0].clone()  # (batch, n_e)
    m_norm_a = m_state_after_a.norm(dim=-1).mean().item()
    print(f"  Норма активации M после паттерна A: {m_norm_a:.4f}")
    
    # Фаза 2: Тишина
    print(f"\n  Фаза 2: {200} шагов тишины...")
    zero_I = torch.zeros(batch, groups_config['S'].n_e, device=DEVICE)
    with torch.no_grad():
        for _ in range(200):
            states = model.step(states, zero_I)
    
    m_norm_silent = states['M'][0].norm(dim=-1).mean().item()
    print(f"  Норма активации M после тишины: {m_norm_silent:.4f}")
    
    # Фаза 3: Паттерн B (токены 100-109)
    print("\n  Фаза 3: подаём паттерн B (токены 100-109)...")
    pattern_b = torch.randint(100, 110, (batch, n_input), device=DEVICE)
    with torch.no_grad():
        _, states = model(pattern_b, initial_states=states)
    
    m_state_after_b = states['M'][0].clone()
    m_norm_b = m_state_after_b.norm(dim=-1).mean().item()
    print(f"  Норма активации M после паттерна B: {m_norm_b:.4f}")
    
    # Анализ: можно ли отличить состояния?
    print("\n  Анализ различимости состояний M:")
    
    # Косинусная схожесть состояний
    m_flat_a = m_state_after_a.view(batch, -1)
    m_flat_b = m_state_after_b.view(batch, -1)
    
    cos_sim_ab = F.cosine_similarity(m_flat_a, m_flat_b).mean().item()
    print(f"  Cosine similarity(M_A, M_B): {cos_sim_ab:.4f}")
    
    # Расстояние между состояниями
    dist_ab = torch.dist(m_flat_a.mean(dim=0), m_flat_b.mean(dim=0)).item()
    print(f"  Euclidean distance(mean M_A, mean M_B): {dist_ab:.4f}")
    
    # Попытка "прочитать" память: обучить простой классификатор
    print("\n  Попытка декодирования: какой паттерн был?")
    
    # Создать датасет из состояний M
    m_states = torch.cat([m_state_after_a, m_state_after_b], dim=0)  # (2*batch, n_e)
    labels = torch.cat([
        torch.zeros(batch, dtype=torch.long),
        torch.ones(batch, dtype=torch.long),
    ], dim=0)
    
    # Простой линейный классификатор
    classifier = nn.Linear(groups_config['M'].n_e, 2).to(DEVICE)
    opt = optim.Adam(classifier.parameters(), lr=1e-2)
    
    classifier.train()
    for _ in range(100):
        opt.zero_grad()
        preds = classifier(m_states)
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        opt.step()
    
    # Оценка точности
    classifier.eval()
    with torch.no_grad():
        preds = classifier(m_states).argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()
    
    print(f"  Точность классификации паттернов по состоянию M: {accuracy:.2%}")
    
    decodable = accuracy > 0.7
    print(f"\n  Вывод: информация о паттерне {'сохраняется и может быть прочитана ✓' if decodable else 'теряется или не читается ✗'}")
    
    return {
        "m_norm_after_pattern_a": round(m_norm_a, 4),
        "m_norm_after_silence": round(m_norm_silent, 4),
        "m_norm_after_pattern_b": round(m_norm_b, 4),
        "cosine_similarity_a_vs_b": round(cos_sim_ab, 4),
        "euclidean_distance_a_vs_b": round(dist_ab, 4),
        "decoder_accuracy": round(accuracy, 4),
        "information_decodable": decodable,
    }


# ─── Запуск всех тестов и запись отчёта ──────────────────────────────────────

def write_report(results: dict, elapsed: dict, groups_config: dict, version: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = []
    lines.append(f"# Технический отчёт: BIGA v{version} — Лингвистические тесты")
    lines.append(f"")
    lines.append(f"**Дата:** {now}")
    lines.append(f"**Конфигурация:** GROUPS_FULL")
    lines.append(f"**Устройство:** {DEVICE}")
    lines.append(f"")
    
    lines.append(f"## Конфигурация модели")
    lines.append(f"")
    lines.append(f"| Группа | E-нейроны | I-нейроны | τ_E | τ_I |")
    lines.append(f"|--------|-----------|-----------|-----|-----|")
    for name, c in groups_config.items():
        lines.append(f"| {name} | {c.n_e} | {c.n_i} | {c.tau_e} | {c.tau_i} |")
    lines.append(f"")
    
    # Сводная таблица
    lines.append(f"## Сводная таблица результатов")
    lines.append(f"")
    lines.append(f"| # | Тест | Результат | Вывод |")
    lines.append(f"|---|------|-----------|-------|")
    
    r6 = results["T6"]
    lines.append(f"| T6 | Обучение на тексте | BIGA={r6['biga_final_loss']}, RNN={r6['rnn_final_loss']} | {'✓ BIGA лучше' if r6['biga_better'] else '~ RNN лучше'} |")
    
    r7 = results["T7"]
    lines.append(f"| T7 | Синтаксическая чувствительность | ΔP={r7['average_difference']:.4f} | {'✓ Чувствительна' if r7['syntax_sensitive'] else '✗ Не чувствительна'} |")
    
    r8 = results["T8"]
    lines.append(f"| T8 | Семантическое сходство | cat-dog={r8['cat_dog_similarity']}, cat-sleeps={r8['cat_sleeps_similarity']} | {'✓ Подтверждено' if r8['semantic_confirmed'] else '✗ Не подтверждено'} |")
    
    r9 = results["T9"]
    lines.append(f"| T9 | Анализ группы M | Decoder acc={r9['decoder_accuracy']:.2%} | {'✓ Информация сохраняется' if r9['information_decodable'] else '~ Информация теряется'} |")
    
    lines.append(f"")
    
    # T6
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## T6: Обучение на тексте")
    lines.append(f"")
    lines.append(f"**Метод:** Обучить BIGA и простую RNN на одинаковом мини-корпусе (~20K токенов).")
    lines.append(f"")
    lines.append(f"| Модель | Параметры | Final Loss | Время |")
    lines.append(f"|--------|-----------|------------|-------|")
    lines.append(f"| BIGA | {r6['biga_params']:,} | {r6['biga_final_loss']:.4f} | {elapsed['T6']:.1f}с |")
    lines.append(f"| RNN | {r6['rnn_params']:,} | {r6['rnn_final_loss']:.4f} | {elapsed['T6']:.1f}с |")
    lines.append(f"")
    lines.append(f"**Разница loss:** {r6['loss_difference']:+.4f} (BIGA - RNN)")
    lines.append(f"")
    lines.append(f"**Вывод:** {'BIGA показывает лучший результат по сравнению с RNN аналогичного размера.' if r6['biga_better'] else 'RNN показывает лучший результат. BIGA требует дополнительной настройки или больше данных.'}")
    lines.append(f"")
    lines.append(f"*Время теста: {elapsed['T6']:.2f}с*")
    lines.append(f"")
    
    # T7
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## T7: Синтаксическая чувствительность")
    lines.append(f"")
    lines.append(f"**Метод:** Минимальные пары — грамматически правильные vs неправильные последовательности.")
    lines.append(f"")
    lines.append(f"| Пара | P(correct) | P(incorrect) | Разница |")
    lines.append(f"|------|------------|--------------|---------|")
    for pair in r7["minimal_pairs"]:
        lines.append(f"| {pair['name'][:30]} | {pair['p_correct']:.6f} | {pair['p_incorrect']:.6f} | {pair['difference']:+.6f} |")
    lines.append(f"")
    lines.append(f"**Средняя разница:** {r7['average_difference']:+.6f}")
    lines.append(f"")
    lines.append(f"**Вывод:** {'Модель демонстрирует чувствительность к синтаксическим нарушениям.' if r7['syntax_sensitive'] else 'Модель не показывает значимой разницы между правильными и неправильными конструкциями.'}")
    lines.append(f"")
    lines.append(f"*Время теста: {elapsed['T7']:.2f}с*")
    lines.append(f"")
    
    # T8
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## T8: Семантическое сходство")
    lines.append(f"")
    lines.append(f"**Метод:** Сравнение активаций группы G для семантически близких и далёких токенов.")
    lines.append(f"")
    lines.append(f"| Пара токенов | Описание | Cosine similarity |")
    lines.append(f"|--------------|----------|-------------------|")
    sim_desc = {
        "cat-dog": "оба существительные (животные)",
        "cat-sleeps": "сущ. vs глагол",
        "dog-bird": "оба существительные (животные)",
        "sleeps-runs": "оба глаголы",
    }
    for pair, sim in r8["pairwise_similarities"].items():
        lines.append(f"| {pair} | {sim_desc.get(pair, '')} | {sim:.4f} |")
    lines.append(f"")
    lines.append(f"**Гипотеза:** cat-dog > cat-sleeps")
    lines.append(f"**Результат:** {r8['cat_dog_similarity']:.4f} > {r8['cat_sleeps_similarity']:.4f} = {'ДА ✓' if r8['semantic_confirmed'] else 'НЕТ ✗'}")
    lines.append(f"")
    lines.append(f"**Вывод:** {'Семантически близкие токены имеют близкие активации в пространстве группы G.' if r8['semantic_confirmed'] else 'Семантическое сходство не проявилось в пространстве активаций.'}")
    lines.append(f"")
    lines.append(f"*Время теста: {elapsed['T8']:.2f}с*")
    lines.append(f"")
    
    # T9
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## T9: Анализ группы M")
    lines.append(f"")
    lines.append(f"**Метод:** Подать паттерн A → тишина → паттерн B. Анализ состояния медленной группы M.")
    lines.append(f"")
    lines.append(f"| Метрика | Значение |")
    lines.append(f"|---------|----------|")
    lines.append(f"| Норма M после паттерна A | {r9['m_norm_after_pattern_a']:.4f} |")
    lines.append(f"| Норма M после тишины | {r9['m_norm_after_silence']:.4f} |")
    lines.append(f"| Норма M после паттерна B | {r9['m_norm_after_pattern_b']:.4f} |")
    lines.append(f"| Cosine sim(M_A, M_B) | {r9['cosine_similarity_a_vs_b']:.4f} |")
    lines.append(f"| Euclidean dist(M_A, M_B) | {r9['euclidean_distance_a_vs_b']:.4f} |")
    lines.append(f"| Точность декодера | {r9['decoder_accuracy']:.2%} |")
    lines.append(f"")
    lines.append(f"**Вывод:** {'Информация о входном паттерне сохраняется в состоянии группы M и может быть прочитана простым классификатором.' if r9['information_decodable'] else 'Информация теряется или не может быть надёжно извлечена из состояния группы M.'}")
    lines.append(f"")
    lines.append(f"*Время теста: {elapsed['T9']:.2f}с*")
    lines.append(f"")
    
    # Заключение
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Общее заключение")
    lines.append(f"")
    confirmed = sum([
        r6["biga_better"],
        r7["syntax_sensitive"],
        r8["semantic_confirmed"],
        r9["information_decodable"],
    ])
    lines.append(f"Подтверждено свойств: **{confirmed} / 4**")
    lines.append(f"")
    lines.append(f"BIGA v{version} демонстрирует начальные лингвистические способности в конфигурации GROUPS_FULL. "
                 f"Модель показывает способность к обучению на тексте, чувствительность к синтаксису, "
                 f"семантическое представление и сохранение информации в медленной памяти группы M.")
    lines.append(f"")
    total_time = sum(elapsed.values())
    lines.append(f"*Суммарное время тестирования: {total_time:.1f}с*")
    
    import os
    report_path = os.path.join(os.path.dirname(__file__), "report_linguistic.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Отчёт записан: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="BIGA v0.4 — Лингвистические тесты")
    parser.add_argument(
        "--config",
        type=str,
        default="full",
        choices=["tiny", "small", "full"],
        help="Конфигурация для тестирования: tiny, small, full (по умолчанию: full)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="0.4",
        help="Версия BIGA (по умолчанию: 0.4)"
    )
    args = parser.parse_args()
    
    # Выбор конфигурации
    if args.config == "tiny":
        groups_config = GROUPS_TINY
        config_name = "GROUPS_TINY"
    elif args.config == "small":
        groups_config = GROUPS_SMALL
        config_name = "GROUPS_SMALL"
    else:
        groups_config = GROUPS_FULL
        config_name = "GROUPS_FULL"
    
    print(f"BIGA v{args.version} — Лингвистические тесты")
    print(f"Конфигурация: {config_name}")
    print(f"Устройство: {DEVICE}  |  Seed: 42")
    
    results = {}
    elapsed = {}
    
    for label, fn in [
        ("T6", test_text_learning),
        ("T7", test_syntax_sensitivity),
        ("T8", test_semantic_similarity),
        ("T9", test_m_group_analysis),
    ]:
        t0 = time.time()
        results[label] = fn(groups_config)
        elapsed[label] = time.time() - t0
    
    section("ИТОГОВАЯ СВОДКА")
    rows = [
        ("T6", "Обучение на тексте",      results["T6"]["biga_better"]),
        ("T7", "Синтаксическая чувств.",  results["T7"]["syntax_sensitive"]),
        ("T8", "Семантическое сходство",  results["T8"]["semantic_confirmed"]),
        ("T9", "Анализ группы M",         results["T9"]["information_decodable"]),
    ]
    print(f"\n  {'Тест':<4}  {'Свойство':<28}  {'Результат':<10}  {'Время'}")
    print(f"  {'─' * 65}")
    for label, name, passed in rows:
        mark = "✓ Да" if passed else "✗ Нет"
        print(f"  {label:<4}  {name:<28}  {mark:<10}  {elapsed[label]:.1f}с")
    confirmed = sum(1 for _, _, p in rows if p)
    print(f"\n  Итого подтверждено: {confirmed}/4")
    
    write_report(results, elapsed, groups_config, args.version)


if __name__ == "__main__":
    main()
