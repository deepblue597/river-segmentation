# river-segmentation

This repo covers all the files necessary for model training, image predictions, databases configuration and dashboards of river segmentation

## Architecture Overview

```mermaid
graph TB
    %% Data Sources
    Camera[📷 Camera] --> Kafka[📨 Kafka<br/>Short-term Storage]

    %% Processing Pipeline
    Kafka --> QuixStreams[🔄 QuixStreams<br/>Processing Pipeline]
    QuixStreams --> Model[🤖 River Segmentation Model<br/>UNet++ EfficientNet-B3]

    %% Storage Systems
    Kafka --> MinIO_Raw[🗄️ MinIO<br/>Raw Image Storage]
    Model --> MinIO_Predictions[🗄️ MinIO<br/>Prediction Overlays]
    Model --> TimescaleDB[🗃️ TimescaleDB<br/>Model Results]

    %% Visualization
    TimescaleDB --> Grafana_Metrics[📊 Grafana<br/>Metrics Dashboard]
    MinIO_Predictions --> Grafana_Images[📊 Grafana<br/>Image Visualization]
    MinIO_Raw --> Grafana_Images

    %% Styling
    classDef storage fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef visualization fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef source fill:#fff3e0,stroke:#f57c00,stroke-width:2px

    class Camera source
    class Kafka,MinIO_Raw,MinIO_Predictions,TimescaleDB storage
    class QuixStreams,Model processing
    class Grafana_Metrics,Grafana_Images visualization
```

## Data Flow Description

### 1. **Data Ingestion**

- **Camera** captures real-time river images
- **Kafka** provides reliable, scalable message streaming for temporary storage

### 2. **Processing Pipeline**

- **QuixStreams** processes Kafka messages in real-time
- **River Segmentation Model** (UNet++ with EfficientNet-B3) performs water detection

### 3. **Storage Architecture**

- **MinIO** (Raw): Stores original camera images
- **MinIO** (Predictions): Stores processed images with prediction overlays
- **TimescaleDB**: Stores model results (water coverage %, confidence scores, overflow detection)

### 4. **Visualization & Monitoring**

- **Grafana Metrics**: Real-time dashboards showing water levels, alerts, trends
- **Grafana Images**: Visual monitoring of predictions and camera feeds

## docker compose configuration

In order to run all our apps in one unified way we are using a docker-compose file.
**NOTE**: you will need to change some settings to run properly

- for the grafana data folder you need to run `sudo chown -R 472:472 data/grafana/`
- for the timescaledb folder you need to run `sudo chown -R 1000:1000 data/timescaledb/`

## Kafka Connection Guide

The Kafka cluster provides multiple connection options for different use cases:

### 🚀 Quick Start

Start the entire pipeline:

```bash
docker-compose up -d
```

Access Kafka UI at: http://localhost:8080

### 📡 Connection Endpoints

#### **Internal Services (Docker Network)**

- **PLAINTEXT**: `broker-riverseg-1:19092,broker-riverseg-2:19092,broker-riverseg-3:19092`
- Used by containers within the same Docker network (like Kafka UI)

#### **External Applications (Local Machine)**

- **PLAINTEXT (No Auth)**: `localhost:29092,localhost:39092,localhost:49092`
- **SASL (With Auth)**: `localhost:29093,localhost:39093,localhost:49093`

#### **External Devices (Remote Access)**

Replace `192.168.1.196` with your host machine's IP address:

- **PLAINTEXT (No Auth)**: `192.168.1.196:29092,192.168.1.196:39092,192.168.1.196:49092`
- **SASL (With Auth)**: `192.168.1.196:29093,192.168.1.196:39093,192.168.1.196:49093`

### 🔐 SASL Authentication

**Credentials:**

- **Admin User**: `admin` / `admin123`
- **Client User**: `client` / `client123`

**Configuration Examples:**

#### Python (kafka-python)

```python
from kafka import KafkaProducer, KafkaConsumer

# Without Authentication
producer = KafkaProducer(
    bootstrap_servers=['localhost:29092', 'localhost:39092', 'localhost:49092']
)

# With SASL Authentication
producer = KafkaProducer(
    bootstrap_servers=['localhost:29093', 'localhost:39093', 'localhost:49093'],
    security_protocol='SASL_PLAINTEXT',
    sasl_mechanism='PLAIN',
    sasl_plain_username='client',
    sasl_plain_password='client123'
)
```

#### Java (Spring Boot)

```yaml
# application.yml - Without Auth
spring:
  kafka:
    bootstrap-servers: localhost:29092,localhost:39092,localhost:49092

# application.yml - With SASL Auth
spring:
  kafka:
    bootstrap-servers: localhost:29093,localhost:39093,localhost:49093
    security:
      protocol: SASL_PLAINTEXT
    sasl:
      mechanism: PLAIN
      jaas:
        config: org.apache.kafka.common.security.plain.PlainLoginModule required username="client" password="client123";
```

#### Node.js (kafkajs)

```javascript
const kafka = require("kafkajs");

// Without Authentication
const client = kafka({
  clientId: "river-segmentation-app",
  brokers: ["localhost:29092", "localhost:39092", "localhost:49092"],
});

// With SASL Authentication
const client = kafka({
  clientId: "river-segmentation-app",
  brokers: ["localhost:29093", "localhost:39093", "localhost:49093"],
  sasl: {
    mechanism: "plain",
    username: "client",
    password: "client123",
  },
});
```

#### CLI Tools (kafka-console)

```bash
# Without Authentication
kafka-console-producer --bootstrap-server localhost:29092 --topic river-images

# With SASL Authentication
kafka-console-producer --bootstrap-server localhost:29093 \
  --producer.config sasl.properties \
  --topic river-images

# sasl.properties content:
# security.protocol=SASL_PLAINTEXT
# sasl.mechanism=PLAIN
# sasl.jaas.config=org.apache.kafka.common.security.plain.PlainLoginModule required username="client" password="client123";
```

### 🌐 Network Configuration

**Finding Your Host IP:**

```bash
# Windows
ipconfig | findstr "IPv4"

# Linux/Mac
ip addr show | grep inet
```

**Firewall Ports:**
Ensure these ports are open for external access:

- `29092, 29093` (Broker 1)
- `39092, 39093` (Broker 2)
- `49092, 49093` (Broker 3)

### 📊 Monitoring & Management

- **Kafka UI**: http://localhost:8080
- **Grafana**: http://localhost:3001
- **MinIO Console**: http://localhost:9001
- **TimescaleDB**: `localhost:5432` (user: postgres)

### 🔧 Common Topics

Based on the river segmentation pipeline:

- `river-images` - Raw camera images
- `prediction-results` - Model output with water coverage data
- `alerts` - Overflow/alert notifications
- `metrics` - Performance and system metrics
