# Legal Contract Processing Pipeline - Docker Image
# ================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY contract_processor.py .
COPY config.py .
COPY example_usage.py .

# Create necessary directories
RUN mkdir -p data outputs logs

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import contract_processor; print('OK')" || exit 1

# Default command
CMD ["python", "contract_processor.py", "--help"]

# Labels for metadata
LABEL maintainer="AI Developer <developer@example.com>"
LABEL version="1.0.0"
LABEL description="Legal Contract Processing Pipeline with LLMs"
LABEL org.opencontainers.image.source="https://github.com/your-repo/legal-contract-processor"
