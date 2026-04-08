FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose ports for OpenEnv and the IDE
EXPOSE 7860

# Launch both servers in the background
CMD ["uvicorn", "src.server:app", "--reload", "--host", "0.0.0.0", "--port", "7860"]