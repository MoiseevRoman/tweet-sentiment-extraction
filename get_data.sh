#!/bin/sh
# Скрипт для загрузки данных через DVC из публичного S3 хранилища
# Данные хранятся в Yandex Object Storage: https://storage.yandexcloud.net/hw2-mlops-data

source .venv/bin/activate 2>/dev/null || true

echo "Загрузка данных через DVC из S3..."
uv run dvc pull

echo "Данные загружены в data/raw/"
