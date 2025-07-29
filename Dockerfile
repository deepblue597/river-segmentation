# Use a more secure base image with latest security patches
FROM python:3.13-slim

# Install system dependencies for OpenCV and security updates
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with security updates
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

#RUN pip install .
# Copy application code and model
COPY . .

# Set environment variables for configuration
ENV MODEL_PATH=best_model.pth.tar
ENV MODEL_NAME=unetplusplus_efficientnet-b3
ENV OVERFLOW_THRESHOLD=80

ENV MINIO_ADDRESS=linux-pc
ENV MINIO_PORT=9000
ENV MINIO_BUCKET=river
ENV MINIO_ACCESS_KEY=minio
ENV MINIO_SECRET_KEY=minio123

ENV TIMESCALE_ADDRESS=linux-pc
ENV TIMESCALE_PORT=5432
ENV TIMESCALE_DB=river
ENV TIMESCALE_USER=postgres
ENV TIMESCALE_PASSWORD=password
ENV TIMESCALE_TABLE=river_segmentation

ENV KAFKA_ADDRESS=linux-pc
ENV KAFKA_PORT=39092
ENV KAFKA_TOPIC=River
ENV KAFKA_CONSUMER_GROUP=model-prediction-06
ENV KAFKA_AUTO_OFFSET_RESET=earliest
ENV KAFKA_SECURITY_PROTOCOL=plaintext

# Create a non-root user for security
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Add healthcheck to verify the application is working
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import torch, cv2, numpy as np; from models.model import RiverSegmentationModel; import os; exit(0 if os.path.exists('best_model.pth.tar') else 1)" || exit 1

# Run the streaming application
CMD ["python", "streaming.py"]
