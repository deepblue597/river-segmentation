from quixstreams import Application
from quixstreams.kafka import ConnectionConfig
import base64
import os

connectionConfig = ConnectionConfig(
    bootstrap_servers="linux-pc:39092", security_protocol="plaintext"
)

app = Application(
    broker_address=connectionConfig,
    consumer_group="consumer",
    auto_offset_reset="earliest",
)


input_topic = app.topic("River", value_deserializer="json")

sdf_stream = app.dataframe(topic=input_topic)

# Create output directory if it doesn't exist
output_dir = "saved_images"
os.makedirs(output_dir, exist_ok=True)


def process_image(row):
    # Print to debug structure if needed
    print(row)

    # Replace 'image' with your actual key if different
    image_data_base64 = row["image"]

    # Decode from base64 to raw bytes
    image_bytes = base64.b64decode(image_data_base64)

    # Create a unique filename, e.g. using Kafka offset or a counter
    filename = os.path.join(output_dir, row["filename"])

    # Save the image
    with open(filename, "wb") as f:
        f.write(image_bytes)

    print(f"Image saved as {filename}")


sdf_stream = sdf_stream.apply(process_image)

if __name__ == "__main__":
    # Start the application
    app.run()
