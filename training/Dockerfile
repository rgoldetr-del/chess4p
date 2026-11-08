# Dockerfile для Chess4P AI Training
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создание директорий для данных
RUN mkdir -p /app/models /app/runs /app/data

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Команда по умолчанию
CMD ["python", "training/model.py"]
