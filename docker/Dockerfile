FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/models/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
# Install CPU-specific PyTorch to save space and avoid pulling CUDA binaries
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY api_server.py .
COPY modules/ ./modules/
COPY config/ ./config/

# Ensure directories exist for volumes
RUN mkdir -p /app/models /app/data /app/logs

# Expose the API port
EXPOSE 11435

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:11435/ || exit 1

# Run the WSGI server
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:11435", "--timeout", "120", "api_server:app"]
