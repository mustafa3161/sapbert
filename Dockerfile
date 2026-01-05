FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV TRANSFORMERS_CACHE=/app/model_cache

EXPOSE 5000

CMD ["python", "app.py"]
