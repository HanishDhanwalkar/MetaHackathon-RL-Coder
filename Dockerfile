FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jinja2  # Required for the editor

COPY . .

# Expose ports for OpenEnv and the IDE
EXPOSE 8000
EXPOSE 8080

# Launch both servers in the background
CMD uvicorn src.server:app --host 0.0.0.1 --port 8000 & \
    uvicorn src.editor:app --host 0.0.0.1 --port 8080