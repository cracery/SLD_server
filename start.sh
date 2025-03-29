#!/bin/sh
# Використовуємо порт із змінної середовища $PORT, або 8000 за замовчуванням
PORT_TO_USE="${PORT:-8000}"

exec gunicorn --bind 0.0.0.0:$PORT_TO_USE main:app --timeout 120 --preload
