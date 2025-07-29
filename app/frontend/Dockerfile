FROM python:3.13-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install .

COPY . .

# Create a non-root user for security
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

USER appuser

ENV BACKEND_IP=localhost
ENV BACKEND_PORT=8000

# Expose the application port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]


