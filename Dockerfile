# TAOS - Tuanhe Aqueduct Autonomous Operation System
# Docker Configuration

FROM python:3.11-slim

LABEL maintainer="TAOS Team"
LABEL version="3.6"
LABEL description="团河渡槽自主运行系统"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TAOS_ENV=production
ENV TAOS_API_PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p /app/src/data /app/src/logs /app/config

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

# Run the application
WORKDIR /app/src
CMD ["python", "server.py"]
