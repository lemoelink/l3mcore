# ==============================================================================
# l3mcore - Lightweight MoE Routing Proxy (Standalone Multi-Stage Dockerfile)
# ==============================================================================

# --- Stage 1: Build virtual environment and dependencies ---
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    find /opt/venv -type f -name '*.pyc' -delete && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} +

# --- Stage 2: Lean Runtime ---
FROM python:3.10-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY api_server.py .
COPY main.py .
COPY modules/ ./modules/
COPY config/ ./config/

RUN mkdir -p /app/models /app/data /app/logs

EXPOSE 11435

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:11435/health || exit 1

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:11435", "--timeout", "120", "api_server:app"]
