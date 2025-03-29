FROM python:3.10-slim
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


RUN pip install gdown


RUN mkdir -p /app/weights && \
    gdown --id 1om2AMju2ywu6QABY5xaM10KMr-4d8hhh -O /app/weights/vgg_face_weights.h5

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/

EXPOSE 8000

CMD exec gunicorn --bind 0.0.0.0:${PORT:-8000} main:app --timeout 120 --preload
