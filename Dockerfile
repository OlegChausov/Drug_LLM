FROM python:3.12.7-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# All Python deps (torch CPU pinned via --extra-index-url in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY LLM_app/ ./LLM_app/
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "LLM_app.main:app", "--host", "0.0.0.0", "--port", "8000"]