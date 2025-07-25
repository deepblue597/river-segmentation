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
    Model --> MinIO_Raw[🗄️ MinIO<br/>Raw Image Storage]
    Model --> MinIO_Predictions[🗄️ MinIO<br/>Prediction Mask]
    Model --> MinIO_Confidence[🗄️ MinIO<br/>Confidence Mask]
    Model --> MinIO_Overlay[🗄️ MinIO<br/>Overlay prediction on image]
    Model --> TimescaleDB[🗃️ TimescaleDB<br/>Model Results]

    %% Visualization
    TimescaleDB --> Grafana_Metrics[📊 Grafana<br/>Metrics Dashboard]


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

## 🏗️ Code Architecture

### 📁 Project Structure

```text
river-segmentation/
├── models/
│   ├── __init__.py
│   └── model.py                    # RiverSegmentationModel class
├── connectors/
│   ├── __init__.py                 # Abstract Connector base class
│   ├── kafka_connector.py          # Kafka/QuixStreams integration
│   ├── minio_connector.py          # MinIO object storage
│   └── timescale_connector.py      # TimescaleDB time-series database
├── prediction_v2.py               # Production pipeline runner
```

### 🔌 Connector Architecture

The project uses a clean, modular connector architecture based on abstract base classes:

#### **Abstract Base Connector** (`connectors/__init__.py`)

```python
from abc import ABC, abstractmethod

class Connector(ABC):
    def __init__(self, address, port, target):
        self.address = address
        self.port = port
        self.target = target

    @abstractmethod
    def connect(self): pass

    @abstractmethod
    def disconnect(self): pass

    @abstractmethod
    def is_connected(self): pass

    @abstractmethod
    def get_connection_info(self): pass
```

#### **Kafka Connector** (`connectors/kafka_connector.py`)

- **Purpose**: Handles QuixStreams/Kafka message consumption
- **Features**:
  - SASL authentication support
  - Consumer group management
  - Stream dataframe creation for real-time processing
  - Automatic reconnection handling

**Usage:**

```python
kafka_conn = KafkaConnector(
    address="localhost",
    port=29092,
    target="river-images",  # topic name
    consumer_group="model-prediction",
    security_protocol="plaintext"
)
kafka_conn.connect()
```

#### **MinIO Connector** (`connectors/minio_connector.py`)

- **Purpose**: S3-compatible object storage for images
- **Features**:
  - Stores raw images, prediction masks, confidence maps, overlays
  - Automatic image encoding (PNG format)
  - Bucket management
  - Connection health checking

**Usage:**

```python
minio_conn = MinIOConnector(
    address="localhost",
    port=9000,
    target="river",  # bucket name
    access_key="minio",
    secret_key="minio123"
)
minio_conn.connect()
```

#### **TimescaleDB Connector** (`connectors/timescale_connector.py`)

- **Purpose**: Time-series database for model metrics and results
- **Features**:
  - Stores water coverage percentages, confidence scores
  - Overflow detection flags
  - Automatic table metadata loading
  - SQL execution with proper escaping

**Usage:**

```python
timescale_conn = TimescaleConnector(
    address="localhost",
    port=5432,
    target="river",  # database name
    username="postgres",
    password="password"
)
timescale_conn.connect()
```

### 🤖 River Segmentation Model (`models/model.py`)

#### **Key Features:**

1. **Flexible Architecture Support:**

   - UNet, UNet++, DeepLabV3+, FPN
   - Configurable encoders (ResNet, EfficientNet, etc.)
   - Custom input sizes and normalization

2. **Complete Pipeline Integration:**

   - Base64 image decoding
   - Preprocessing with Albumentations
   - Model inference with confidence mapping
   - Result encoding and storage

3. **Production-Ready Features:**
   - Overflow detection with configurable thresholds
   - Error handling and logging
   - Stream processing integration
   - Device auto-detection (CPU/GPU)

#### **Model Configuration:**

```python
model = RiverSegmentationModel(
    model_path="best_model.pth.tar",
    model_name="unetplusplus_efficientnet-b3",
    model_architecture="unetplusplus",  # unet, deeplabv3plus, fpn
    encoder_name="efficientnet-b3",     # any timm encoder
    encoder_weights="imagenet",
    input_size=(512, 512),
    overflow_threshold=80,              # % threshold for overflow
    mean=(0.485, 0.456, 0.406),        # ImageNet normalization
    std=(0.229, 0.224, 0.225)
)
```

#### **Prediction Pipeline:**

1. **Image Processing:**

   ```text
   Base64 → Bytes → NumPy → OpenCV → RGB → Resize → Normalize → Tensor
   ```

2. **Model Inference:**

   ```text
   Input Tensor → Model → Sigmoid → Confidence Map → Binary Mask
   ```

3. **Results Storage:**

   ```text
   Prediction Mask → MinIO (predictions/)
   Confidence Map → MinIO (confidence_maps/)
   Overlay Image → MinIO (overlays/)
   Statistics → TimescaleDB (predictions table)
   ```

#### **Output Data Structure:**

The model returns comprehensive prediction results:

```python
{
    'filename': 'image_001.jpg',
    'water_coverage': 23.5,          # Percentage of water pixels
    'avg_confidence': 0.87,          # Average model confidence
    'overflow_flag': False,          # True if > threshold
    'confidence_map': numpy_array,   # Raw confidence scores
    'prediction_mask': numpy_array,  # Binary water mask
    'overlay_image': bytes          # Encoded overlay image
}
```

### 🚀 Production Pipeline (`prediction_v2.py`)

The main production script that orchestrates the entire real-time prediction pipeline:

#### **Pipeline Flow:**

1. **Initialization:**

   - Load trained model weights
   - Initialize model architecture
   - Setup device (GPU/CPU)

2. **Service Connections:**

   - Connect to Kafka for message consumption
   - Connect to MinIO for image storage
   - Connect to TimescaleDB for metrics storage

3. **Stream Processing:**
   - Apply prediction function to Kafka stream
   - Process images in real-time
   - Store results automatically

#### **Usage:**

```bash
# Start the production pipeline
python prediction_v2.py
```

#### **Error Handling:**

- Automatic reconnection for failed connections
- Graceful degradation if services are unavailable
- Comprehensive logging for debugging
- Message processing continues on individual failures

#### **Monitoring:**

- Real-time console output showing:
  - Water coverage percentages
  - Model confidence scores
  - Processing times
  - Connection status
  - Error notifications

## docker compose configuration

In order to run all our apps in one unified way we are using a docker-compose file.
**NOTE**: you will need to change some settings to run properly for linux

- for the grafana data folder you need to run `sudo chown -R 472:472 data/grafana/`
- for the timescaledb folder you need to run `sudo chown -R 1000:1000 data/timescaledb/`

## Kafka Connection Guide

The Kafka cluster provides multiple connection options for different use cases:

### 🚀 Quick Start

Start the entire pipeline:

```bash
docker-compose up -d
```

Access Kafka UI at: <http://localhost:8080>

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

- **Kafka UI**: <http://localhost:8080>
- **Grafana**: <http://localhost:3001>
- **MinIO Console**: <http://localhost:9001>
- **TimescaleDB**: `localhost:5432` (user: postgres)

### 🔧 Common Topics

Based on the river segmentation pipeline:

- `river-images` - Raw camera images
- `prediction-results` - Model output with water coverage data
- `alerts` - Overflow/alert notifications
- `metrics` - Performance and system metrics
