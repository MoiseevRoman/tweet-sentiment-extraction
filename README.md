# Tweet Sentiment Span Extraction

## Описание проекта

Проект решает задачу извлечения подстроки (span) из твита по заданной тональности (positive/negative/neutral) на основе соревнования Kaggle "Tweet Sentiment Extraction".

### Постановка задачи

Дано:
- Текст твита (`text`)
- Тональность твита (`sentiment`: positive/negative/neutral)

Требуется:
- Найти подстроку в тексте, которая наиболее точно отражает заданную тональность (`selected_text`)

### Формат входных и выходных данных

**Входные данные (train.csv):**
- `textID` - идентификатор твита
- `text` - полный текст твита
- `sentiment` - тональность (positive/negative/neutral)
- `selected_text` - целевая подстрока (только для обучения)

**Выходные данные (inference):**
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

### Валидация и тестирование

- Валидация: случайное разбиение обучающей выборки (80/20)
- Тестирование: на тестовом наборе Kaggle

### Датасет

Датасет взят из соревнования Kaggle: [Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction)

### Модель

Архитектура:
- Backbone: RoBERTa-base (transformer encoder)
- Head: Linear(hidden_size * 2 → 2) для предсказания start/end индексов
- Используются последние два слоя hidden states для лучшего качества

**Baseline модель:**
- Упрощенная версия: использует только последний слой hidden states (вместо двух)
- Для обучения baseline измените `use_last_two_layers: false` в `config.yaml`

Обучение:
- Loss: CrossEntropy для start и end позиций
- Optimizer: AdamW с weight decay
- Scheduler: Cosine с warmup (опционально)

### Внедрение

Модель может быть использована для:
- Анализа тональности в социальных сетях
- Извлечения ключевых фраз из текста
- Понимания причин определенной тональности

---

## Быстрый старт

### Полная инструкция по запуску модели

#### Шаг 1: Установка зависимостей

```bash
# 1. Клонировать репозиторий
git clone https://github.com/MoiseevRoman/tweet-sentiment-extraction.git
cd tweet-sentiment-extraction

# 2. Установить uv (если еще не установлен)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Установить зависимости
uv sync

# 4. Установить pre-commit хуки
pre-commit install
```

#### Шаг 2: Загрузка данных

```bash
# Загрузить данные через DVC из S3 хранилища
./get_data.sh

# Проверить, что данные загружены
ls -la data/raw/
# Должны быть: train.csv и test.csv
```

**Примечание:** Если DVC недоступен, данные будут автоматически загружены через Kaggle API при первом запуске обучения (требуются переменные окружения `KAGGLE_USERNAME` и `KAGGLE_KEY`).

#### Шаг 3: Запуск MLflow (опционально, для мониторинга обучения)

```bash
# Запустить MLflow и PostgreSQL через docker-compose
docker compose up -d mlflow postgres

# MLflow будет доступен по адресу: http://localhost:8080
```

#### Шаг 4: Обучение модели

**Вариант A: Debug режим (CPU, быстрая проверка)**

```bash
uv run main.py mode=train \
    train_config.data_config.max_samples=1500 \
    train_config.num_epochs=2 \
    train_config.accelerator=cpu \
    train_config.precision=32
```

**Вариант B: Полное обучение (GPU)**

```bash
uv run main.py mode=train \
    train_config.accelerator=gpu \
    train_config.precision=16-mixed
```

**Вариант C: Обучение с кастомными параметрами**

```bash
uv run main.py mode=train \
    train_config.learning_rate=5e-5 \
    train_config.num_epochs=10 \
    train_config.data_config.batch_size=32
```

После обучения модель будет сохранена в `outputs/` с именем вида `sentiment_span_extractor_0.XXXX.ckpt`.

#### Шаг 5: Тестирование модели

```bash
# Заменить на имя вашей обученной модели
uv run main.py mode=test \
    test_config.checkpoint=sentiment_span_extractor_0.7000.ckpt
```

#### Шаг 6: Инференс на новых данных

```bash
# Создать входной CSV файл с колонками: text, sentiment (и опционально textID)
# Пример: input.csv
# text,sentiment
# "I love this product!",positive
# "This is terrible",negative

# Выполнить инференс
uv run main.py mode=infer \
    infer_config.checkpoint=sentiment_span_extractor_0.7000.ckpt \
    infer_config.input_csv=input.csv \
    infer_config.output_csv=outputs/predictions.csv

# Результаты будут в outputs/predictions.csv
```

---

## Техническая инструкция

### Установка

#### Предварительные требования
- Python >= 3.9
- uv (менеджер зависимостей)
- Docker и Docker Compose (для MLflow и Triton, опционально)

#### Детальные шаги установки

1. **Клонировать репозиторий:**
```bash
git clone https://github.com/MoiseevRoman/tweet-sentiment-span-extraction.git
cd tweet-sentiment-span-extraction
```

2. **Установить uv** (если еще не установлен):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Установить зависимости:**
```bash
uv sync
```

4. **Установить pre-commit хуки** (рекомендуется):
```bash
pre-commit install
uv run --with pre-commit pre-commit run --all-files
```

Ожидается, что все проверки пройдут успешно (зеленый результат).

### Работа с данными

Проект использует DVC для управления данными. Данные хранятся в публичном S3 хранилище (Yandex Object Storage).

**Загрузка данных:**
```bash
./get_data.sh
```

Или вручную через DVC:
```bash
uv run dvc pull
```

Данные автоматически загружаются из: `https://storage.yandexcloud.net/hw2-mlops-data`

**Примечание:** Если DVC недоступен, данные будут автоматически загружены через Kaggle API при первом запуске обучения (требуются переменные окружения `KAGGLE_USERNAME` и `KAGGLE_KEY`).

### Обучение

#### Debug режим (CPU, быстрая проверка)

Для быстрой проверки работоспособности на небольшом датасете:

```bash
uv run main.py mode=train \
    train_config.data_config.max_samples=1500 \
    train_config.num_epochs=2 \
    train_config.accelerator=cpu \
    train_config.precision=32
```

**Ожидается:**
- Лосс должен убывать
- В MLflow логируются: `train_loss`, `val_loss`, `val_jaccard`, `lr`
- Логируются все гиперпараметры и git commit id

#### Полное обучение (GPU)

```bash
uv run main.py mode=train \
    train_config.accelerator=gpu \
    train_config.precision=16-mixed
```

#### Baseline модель

Для обучения упрощенной baseline модели измените в `config.yaml`:
```yaml
train_config:
  model_config:
    use_last_two_layers: false  # Использует только последний слой
```

#### Настройка параметров обучения

Все параметры настраиваются в `config.yaml`. Можно переопределять через CLI:

```bash
uv run main.py mode=train \
    train_config.learning_rate=5e-5 \
    train_config.num_epochs=10 \
    train_config.data_config.batch_size=32 \
    train_config.data_config.max_samples=5000
```

### Инференс

Для предсказания на новых данных:

```bash
uv run main.py mode=infer \
    infer_config.checkpoint=sentiment_span_extractor_0.7000.ckpt \
    infer_config.input_csv=path/to/input.csv \
    infer_config.output_csv=outputs/pred.csv
```

**Формат входного CSV:**
- `textID` (опционально)
- `text` - текст твита
- `sentiment` - тональность (positive/negative/neutral)

**Формат выходного CSV:**
- `textID`
- `selected_text` - предсказанная подстрока
- `start` - начальный индекс
- `end` - конечный индекс

### Тестирование

Для тестирования обученной модели на тестовом датасете:

```bash
uv run main.py mode=test \
    test_config.checkpoint=sentiment_span_extractor_0.7000.ckpt
```

### Конвертация в ONNX

Для конвертации модели в формат ONNX:

```bash
uv run main.py mode=convert \
    convert_config.checkpoint=sentiment_span_extractor_0.7000.ckpt \
    convert_config.output_path=triton/models/sentiment_span_extractor/1/model.onnx
```

Если `--output-path` не указан, модель будет сохранена в `triton/models/sentiment_span_extractor/1/model.onnx`.

### Инференс через Triton

Для инференса через Triton Inference Server:

1. **Конвертировать модель в ONNX** (см. выше)

2. **Запустить Triton:**
```bash
docker compose up -d triton
```

3. **Выполнить инференс:**
```bash
uv run main.py mode=infer-triton \
    infer_triton_config.input_csv=path/to/input.csv \
    infer_triton_config.output_csv=outputs/pred.csv \
    infer_triton_config.triton_url=http://localhost:8000 \
    infer_triton_config.model_name=sentiment_span_extractor
```

### MLflow

Для просмотра метрик и экспериментов:

1. **Запустить MLflow через docker-compose:**
```bash
docker compose up -d mlflow postgres
```

2. **Открыть в браузере:** http://localhost:8080

**В MLflow логируются:**
- Метрики: `train_loss`, `val_loss`, `val_jaccard`, `lr`
- Гиперпараметры (все из конфигов)
- Git commit ID

### Использование Hydra

Все команды используют **Hydra** для управления конфигурацией. Есть два способа запуска:

1. **Через main.py (рекомендуется):**
   ```bash
   uv run main.py mode=train train_config.data_config.max_samples=1500
   ```

2. **Напрямую через скрипты:**
   ```bash
   uv run sentiment_span_extractor/train.py train_config.data_config.max_samples=1500
   ```

Все параметры настраиваются в `config.yaml` и могут быть переопределены через CLI используя синтаксис Hydra: `section.parameter=value`.

### Структура проекта

```
tweet-sentiment-span-extraction/
├── main.py                        # Точка входа (использует Hydra)
├── config.yaml                    # Главный конфиг Hydra
├── sentiment_span_extractor/     # Основной пакет
│   ├── train.py                   # Скрипт для тренировки
│   ├── infer.py                   # Скрипт для инференса
│   ├── test.py                    # Скрипт для тестирования
│   ├── convert.py                 # Конвертация в ONNX
│   ├── infer_triton.py            # Инференс через Triton
│   ├── data/                      # Модули данных
│   │   ├── datamodule.py         # LightningDataModule
│   │   ├── dataset.py             # Dataset с токенизацией
│   │   ├── preprocessing.py       # Препроцессинг
│   │   └── download.py            # Kaggle download
│   ├── models/                    # Модели
│   │   ├── heads.py               # QA head
│   │   └── span_module.py         # LightningModule
│   ├── metrics/                   # Метрики
│   │   └── jaccard.py             # Jaccard метрика
│   └── utils/                     # Утилиты
│       ├── seed.py                # Установка seed для воспроизводимости
│       ├── logging.py             # MLflow логирование
│       └── dvc_utils.py           # DVC/Kaggle fallback
├── data/                          # Данные
│   ├── raw/                       # Исходные данные
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/                 # Обработанные данные
├── outputs/                       # Модели и чекпоинты
├── tests/                         # Тесты
│   └── test_jaccard.py
├── build/                         # Docker образы
│   └── mlflow_build/
│       └── Dockerfile
├── triton/                        # Triton модели
│   └── models/
│       └── sentiment_span_extractor/
│           ├── 1/
│           └── config.pbtxt
├── dvc.yaml                       # DVC стейджи
├── docker-compose.yaml            # Docker Compose конфигурация
├── get_data.sh                    # Скрипт загрузки данных
├── pyproject.toml                 # Зависимости и инструменты
├── Makefile                       # Make команды
├── .gitignore
├── .dvcignore
└── README.md
```

### Разработка

#### Запуск тестов

```bash
uv run --with pytest pytest tests/ -v
```

Или через Make:
```bash
make test
```

#### Линтинг и форматирование

```bash
# Проверка
uv run --with pre-commit pre-commit run --all-files

# Автофикс
ruff check --fix .
prettier --write "**/*.{md,json,yaml}"
```

Или через Make:
```bash
make lint      # Проверка
make format    # Автофикс
```

#### Make команды

```bash
make install       # Установка зависимостей
make setup         # Установка + pre-commit
make test          # Запуск тестов
make lint          # Линтинг
make format        # Форматирование
make clean         # Очистка кэшей
make train         # Обучение модели
make test-model    # Тестирование модели (нужно указать --checkpoint в Makefile)
make infer         # Инференс (нужно указать параметры в Makefile)
make convert       # Конвертация в ONNX (нужно указать --checkpoint в Makefile)
make infer-triton  # Инференс через Triton (нужно указать параметры в Makefile)
make download      # Загрузка данных
make debug-train   # Debug обучение (CPU, небольшой датасет)
```

### Примечания

- Данные и артефакты (модели, чекпоинты) не коммитятся в git (см. `.gitignore`)
- Используется DVC для управления данными из публичного S3 хранилища
- Все гиперпараметры настраиваются в `config.yaml`
- Код следует PEP 8 и проверяется pre-commit хуками
- Для запуска MLflow и Triton используйте `docker compose up -d`
- При первом запуске обучения данные автоматически загружаются, если их нет
