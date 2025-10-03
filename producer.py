from datetime import datetime
import os
import json
from db_connectors import KafkaProducerConnector

KAFKA_ADDRESS = os.environ.get("KAFKA_ADDRESS", "localhost")
KAFKA_PORT = int(os.environ.get("KAFKA_PORT", 39092))
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "River")
KAFKA_ACKS = os.environ.get("KAFKA_ACKS", "all")
KAFKA_COMPRESSION_TYPE = os.environ.get("KAFKA_COMPRESSION_TYPE", "snappy")

kafkaClient = KafkaProducerConnector(
    address=KAFKA_ADDRESS,
    port=KAFKA_PORT,
    topic=KAFKA_TOPIC,
    acks=KAFKA_ACKS,
    compression_type=KAFKA_COMPRESSION_TYPE,
)


def connect(): 
    try: 
        kafkaClient.connect()
        print("Connected to Kafka successfully.")
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
def produce():
    try: 
        message = { 
            "folder_link": "test_data" 
        }
        timestamp = str(int(datetime.now().timestamp() * 1000))  # milliseconds for uniqueness
        
        kafkaClient.produce(
            key = timestamp,
            value = json.dumps(message) 
        )
        
    except Exception as e:
        print(f"Error producing message: {e}")
        
        
if __name__ == "__main__":
    connect()
    produce()
    kafkaClient.disconnect()
    
    
    