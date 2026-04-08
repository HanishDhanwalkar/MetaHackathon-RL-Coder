FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for OpenEnv and the IDE
EXPOSE 7860

# Production start command for HF Spaces
CMD ["sh", "-c", "uvicorn src.server:app --host 0.0.0.0 --port ${PORT:-7860}"]