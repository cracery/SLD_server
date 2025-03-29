FROM python:3.10-slim

# Встановлюємо системні бібліотеки для OpenCV
RUN apt-get update && apt-get install -y \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxfixes3 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
 && rm -rf /var/lib/apt/lists/*

# Встановлюємо gdown для завантаження ваг з Google Drive
RUN pip install gdown

# Завантажуємо ваги VGGFace з Google Drive (файл має бути публічним!)
RUN mkdir -p /app/weights && \
    gdown 1om2AMju2ywu6QABY5xaM10KMr-4d8hhh -O /app/weights/vgg_face_weights.h5

# Створюємо робочу директорію
WORKDIR /app

# Встановлюємо Python залежності
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код
COPY . /app/

# Порт для Railway
EXPOSE 8000

# Запуск Gunicorn
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8000} main:app --timeout 120 --preload
