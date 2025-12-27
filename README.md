# Tweet Sentiment Span Extraction

## Описание проекта

Проект решает задачу извлечения подстроки (span) из твита по заданной тональности (positive/negative/neutral) на основе соревнования Kaggle "Tweet Sentiment Extraction".

### Постановка задачи

Дано:

- Текст твита (`text`)
- Тональность твита (`sentiment`: positive/negative/neutral)

Требуется найти подстроку в тексте, которая наиболее точно отражает заданную тональность (`selected_text`).

### Формат данных

**Входные данные (train.csv):**

- `textID` - идентификатор твита
- `text` - полный текст твита
- `sentiment` - тональность (positive/negative/neutral)
- `selected_text` - целевая подстрока (только для обучения)

**Выходные данные:**

- `textID` - идентификатор твита
- `selected_text` - предсказанная подстрока
- `start` - начальный индекс спана
- `end` - конечный индекс спана

### Метрики

Основная метрика - **Jaccard Score** (коэффициент Жаккара):

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

где A и B - множества слов в предсказанной и целевой подстроках соответственно.

### Датасет

Датасет взят из соревнования Kaggle: [Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction)

### Модель

**Архитектура:**

- Backbone: RoBERTa-base (transformer encoder)
- Head: Linear(hidden_size × 2 → 2) для предсказания start/end индексов
- Используются последние два слоя hidden states для лучшего качества

**Baseline модель:**

- Упрощенная версия: использует только последний слой hidden states (вместо двух)
- Для обучения baseline измените `use_last_two_layers: false` в `config.yaml`

**Обучение:**

- Loss: CrossEntropy для start и end позиций
- Optimizer: AdamW с weight decay
- Scheduler: Cosine с warmup (опционально)

---

## Установка и запуск

### Предварительные требования

- Python ≥ 3.9
- Docker и Docker Compose (для MLflow и Triton, опционально)

### 1. Установка проекта

```bash
# Клонирование репозитория
git clone https://github.com/MoiseevRoman/tweet-sentiment-span-extraction.git
cd tweet-sentiment-span-extraction

# Установка uv (менеджер зависимостей)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Установка зависимостей
uv sync

# Установка pre-commit хуков (рекомендуется)
pre-commit install
uv run --with pre-commit pre-commit run --all-files
```

### 2. Загрузка данных

```bash
# Автоматическая загрузка данных через DVC
./get_data.sh

# Проверка наличия данных
ls -la data/raw/
# Должны быть: train.csv и test.csv
```

**Примечание:** Если DVC недоступен, данные будут автоматически загружены через Kaggle API при первом запуске обучения (требуются переменные окружения `KAGGLE_USERNAME` и `KAGGLE_KEY`).

### 3. Запуск MLflow (опционально)

```bash
docker compose up -d mlflow postgres
```

MLflow будет доступен по адресу: http://localhost:8080

### 4. Обучение модели

#### Быстрая проверка (CPU)

```bash
uv run main.py mode=train \
    train_config.data_config.max_samples=1500 \
    train_config.num_epochs=2 \
    train_config.accelerator=cpu \
    train_config.precision=32
```

#### Полное обучение (GPU)

```bash
uv run main.py mode=train \
    train_config.accelerator=gpu \
    train_config.precision=16-mixed
```

#### Настройка параметров

Все параметры можно настроить через CLI:

```bash
uv run main.py mode=train \
    train_config.learning_rate=5e-5 \
    train_config.num_epochs=10 \
    train_config.data_config.batch_size=32
```

### 5. Тестирование и инференс

#### Тестирование модели

```bash
uv run main.py mode=test \
    test_config.checkpoint=sentiment_span_extractor_0.7000.ckpt
```

#### Инференс на новых данных

```bash
# Создайте input.csv с колонками: text, sentiment (и опционально textID)
uv run main.py mode=infer \
    infer_config.checkpoint=sentiment_span_extractor_0.7000.ckpt \
    infer_config.input_csv=input.csv \
    infer_config.output_csv=outputs/predictions.csv
```

#### Конвертация в ONNX

```bash
uv run main.py mode=convert \
    convert_config.checkpoint=sentiment_span_extractor_0.7000.ckpt
```

#### Инференс через Triton

1. Запустите Triton:

```bash
docker compose up -d triton
```

2. Выполните инференс:

```bash
uv run main.py mode=infer-triton \
    infer_triton_config.input_csv=input.csv \
    infer_triton_config.output_csv=outputs/pred.csv
```

---

## Технические детали

### Управление конфигурацией

Проект использует Hydra для управления конфигурацией. Все параметры настраиваются в `config.yaml` и могут быть переопределены через CLI:

```bash
uv run main.py mode=train train_config.num_epochs=5
```

### Структура проекта

```
├── main.py                      # Точка входа
├── config.yaml                  # Главный конфиг
├── sentiment_span_extractor/   # Основной пакет
│   ├── train.py                # Скрипт для тренировки
│   ├── infer.py                # Скрипт для инференса
│   ├── test.py                 # Скрипт для тестирования
│   ├── convert.py              # Конвертация в ONNX
│   ├── infer_triton.py         # Инференс через Triton
│   ├── data/                   # Модули данных
│   ├── models/                 # Модели
│   ├── metrics/                # Метрики
│   └── utils/                  # Утилиты
├── data/                       # Данные
├── outputs/                    # Модели и чекпоинты
├── triton/                     # Triton модели
├── dvc.yaml                    # DVC стейджи
├── docker-compose.yaml         # Docker Compose конфигурация
├── get_data.sh                 # Скрипт загрузки данных
└── Makefile                    # Make команды
```

### Полезные Make команды

```bash
make install       # Установка зависимостей
make setup         # Установка + pre-commit
make test          # Запуск тестов
make lint          # Линтинг
make format        # Форматирование
make clean         # Очистка кэшей
make train         # Обучение модели
make debug-train   # Debug обучение (CPU, небольшой датасет)
```

### Особенности реализации

- Данные и артефакты не коммитятся в git (см. `.gitignore`)
- Используется DVC для управления данными из публичного S3 хранилища
- Все гиперпараметры настраиваются в `config.yaml`
- Код следует PEP 8 и проверяется pre-commit хуками
- При первом запуске обучения данные автоматически загружаются, если их нет

---

## Внедрение

Модель может быть использована для:

- Анализа тональности в социальных сетях
- Извлечения ключевых фраз из текста
- Понимания причин определенной тональности

Для мониторинга экспериментов используйте MLflow (доступен по адресу http://localhost:8080 после запуска `docker compose up -d mlflow postgres`). В MLflow логируются:

- Метрики: `train_loss`, `val_loss`, `val_jaccard`, `lr`
- Гиперпараметры (все из конфигов)
- Git commit ID
