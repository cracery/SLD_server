FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
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

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/weights && \
    wget -O /app/weights/vgg_face_weights.h5 https://github.com/rcmalli/keras-vggface/releases/download/v2.0/vgg_face_weights.h5

COPY . /app/

# Дозволяємо виконання скрипту
RUN chmod +x /app/start.sh

EXPOSE 8000
CMD ["/app/start.sh"]
