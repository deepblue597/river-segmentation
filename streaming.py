
from models.model import RiverSegmentationModel
import signal
import sys
import os


model = RiverSegmentationModel(
    model_path=os.environ.get("MODEL_PATH", "best_model.pth.tar"),
    model_name=os.environ.get("MODEL_NAME", "unetplusplus_efficientnet-b3"),
    overflow_threshold=int(os.environ.get("OVERFLOW_THRESHOLD", "80")),
)

model.load_model()

model.minIOConnection(
    address=os.environ.get("MINIO_ADDRESS", "localhost"), 
    port=int(os.environ.get("MINIO_PORT", "9000")), 
    target=os.environ.get("MINIO_BUCKET", "river"),  # Added missing bucket name
    access_key=os.environ.get("MINIO_ACCESS_KEY", "minio"),
    secret_key=os.environ.get("MINIO_SECRET_KEY", "minio123")
)

model.timescaleConnection(
    address=os.environ.get("TIMESCALE_ADDRESS", "localhost"), 
    port=int(os.environ.get("TIMESCALE_PORT", "5432")),
    target=os.environ.get("TIMESCALE_DB", "river"),  # Changed from 'database' to 'target'
    username=os.environ.get("TIMESCALE_USER", "postgres"),
    password=os.environ.get("TIMESCALE_PASSWORD", "password"), 
    table_name=os.environ.get("TIMESCALE_TABLE", "river_segmentation"),
)

model.kafkaConnection(
    address=os.environ.get("KAFKA_ADDRESS", "localhost"),
    port=int(os.environ.get("KAFKA_PORT", "39092")),
    topic=os.environ.get("KAFKA_TOPIC", "River"),  # Changed from 'target' to 'topic'
    consumer_group=os.environ.get("KAFKA_CONSUMER_GROUP", "model-prediction-07"),
    auto_offset_reset=os.environ.get("KAFKA_AUTO_OFFSET_RESET", "earliest"),
    security_protocol=os.environ.get("KAFKA_SECURITY_PROTOCOL", "plaintext")
)

print(f"Model loaded: {model}")

def graceful_shutdown(signum, frame):
    """Handle graceful shutdown when Ctrl+C is pressed"""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    model.disconnect_all()
    print("Exiting...")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, graceful_shutdown)   # Ctrl+C
signal.signal(signal.SIGTERM, graceful_shutdown)  # Termination signal

if __name__ == "__main__":
    try:
        print("Starting model prediction application...")
        print("Press Ctrl+C to stop gracefully")
        # Start the model prediction application
        model.start_streaming()
    except KeyboardInterrupt:
        # This will catch Ctrl+C when the Quix app stops
        print("\nKeyboardInterrupt caught, shutting down...")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        # This ensures cleanup happens no matter how the app exits
        print("Performing cleanup...")
        model.disconnect_all()   