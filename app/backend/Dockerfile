FROM python:3.13-slim

WORKDIR /app

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


COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

#RUN pip install .

COPY . .

# Create a non-root user for security
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

USER appuser

# Set environment variables for configuration
# Example Dockerfile additions
# Do not use hardcoded values in production; use environment variables or secrets management
ENV MINIO_ADDRESS=localhost
ENV MINIO_PORT=9000
ENV MINIO_ACCESS_KEY=minio
ENV MINIO_SECRET_KEY=minio123
ENV MINIO_BUCKET=river

ENV KAFKA_ADDRESS=localhost
ENV KAFKA_PORT=39092
ENV KAFKA_TOPIC=River
ENV KAFKA_ACKS=all
ENV KAFKA_COMPRESSION_TYPE=snappy

EXPOSE 8000

# Run the application 
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]

