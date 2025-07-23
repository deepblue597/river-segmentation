
from models.model import RiverSegmentationModel


model = RiverSegmentationModel(
    model_path="best_model.pth.tar",
    model_name="unetplusplus_efficientnet-b3",
    overflow_threshold=80,
)

model.load_model()

model.minIOConnection(
    address='linux-pc', 
    port=9000, 
    target='river',  # Added missing bucket name
    access_key='minio',
    secret_key='minio123'
)

model.timescaleConnection(
    address='linux-pc', 
    port=5432,
    target='river',  # Changed from 'database' to 'target'
    username='postgres',
    password='password'
)

model.kafkaConnection(
    address='linux-pc',
    port=39092,
    topic='River',  # Changed from 'target' to 'topic'
    consumer_group='model-prediction-04',
    auto_offset_reset='earliest',
    security_protocol='plaintext'
)

print(f"Model loaded: {model}")

if __name__ == "__main__":
    # Start the model prediction application
    model.start_streaming()   