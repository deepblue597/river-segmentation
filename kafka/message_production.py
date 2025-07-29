# %%
import json

# import kagglehub
from producer import create_kafka_producer, delivery_callback
from server_funcs import parse_command_line_arguments
import os
import base64
import glob
from datetime import datetime
# %%


def convert_keys_to_underscores(data):
    """
    Recursively converts all dictionary keys by replacing spaces with underscores.
    """
    if isinstance(data, dict):
        return {
            key.replace(" ", "_").lower(): convert_keys_to_underscores(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [convert_keys_to_underscores(item) for item in data]
    else:
        return data


if __name__ == "__main__":
    args = parse_command_line_arguments()

    # init producer - using PLAINTEXT (no authentication)
    producer = create_kafka_producer(
        bootstrap_server=args.bootstrap_server, acks="all", compression_type="snappy"
    )

    print("Messages are being published to Kafka topic")
    messages_count = 0

    # Path to save_images folder
    images_folder = "saved_images"

    # Check if folder exists
    if not os.path.exists(images_folder):
        print(f"Error: {images_folder} folder not found!")
        exit(1)

    # Get all image files from the folder
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []

    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder, extension)))
        # image_files.extend(glob.glob(os.path.join(images_folder, extension.upper())))

    print(f"Found {len(image_files)} images in {images_folder} folder")

    if len(image_files) == 0:
        print(f"No image files found in {images_folder} folder!")
        exit(1)

    # Send each image to Kafka
    for idx, image_path in enumerate(image_files):
        try:
            # Read image file
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            # Encode to base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Get filename without path
            filename = os.path.basename(image_path)

            # Create message
            message = {
                "filename": filename,
                "image": image_base64,
                "date": datetime.now().isoformat(),
                "message_id": idx,
                "file_size": len(image_data),
            }

            # Send to Kafka
            producer.produce(
                args.topic_name,
                value=json.dumps(message),
                key=str(idx),
                callback=delivery_callback,
            )

            # Poll to handle responses
            producer.poll(0)

            messages_count += 1
            print(
                f"Sent image {idx + 1}/{len(image_files)}: {filename} ({len(image_data)} bytes)"
            )

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # Flush to ensure all messages are sent before exit
    producer.flush()
    print(
        f"Successfully sent {messages_count} images to Kafka topic '{args.topic_name}'"
    )
