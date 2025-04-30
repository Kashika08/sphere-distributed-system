FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["bash", "-c", "python proactive_autoscaler.py"]
