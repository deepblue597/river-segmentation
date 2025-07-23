
from models.model import RiverSegmentationModel
import signal
import sys


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
    password='password', 
    table_name='river_segmentation',
)

model.kafkaConnection(
    address='linux-pc',
    port=39092,
    topic='River',  # Changed from 'target' to 'topic'
    consumer_group='model-prediction-06',
    auto_offset_reset='earliest',
    security_protocol='plaintext'
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