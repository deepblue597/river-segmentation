from quixstreams import Application
import boto3
import json
import base64
from datetime import datetime


# MinIO S3 client configuration
s3_client = boto3.client(
    's3',
    endpoint_url='http://linux-pc:9000',
    aws_access_key_id='minio',
    aws_secret_access_key='minio123',
    region_name='eu-west-2'
)

def save_image_to_minio(message_value):
    """Extract base64 image and save as actual image file in MinIO"""
    try:
        # Parse the JSON message
        image_data = message_value['image']
        key = message_value['filename']
        # Decode base64 to binary
        image_bytes = base64.b64decode(image_data)
        
        # Upload to MinIO as actual image file
        s3_client.put_object(
            Bucket='river',
            Key=key,
            Body=image_bytes,
            ContentType='image/png'  # Adjust based on your image type
        )
        
        print(f"📸 Image saved to MinIO: images/{key}")
        return message_value
    except Exception as e:
        print(f"❌ Error saving image: {e}")
        return message_value

app = Application(
    broker_address='linux-pc:39092',
    consumer_group='minio-fresh', 
    auto_offset_reset='earliest',
    # Add debug logging
    loglevel='DEBUG'
)
topic = app.topic('River')

sdf = app.dataframe(topic=topic)

# Add some debugging
sdf = sdf.update(lambda value: print(f"📥 Received message: {value}") or value)
sdf = sdf.update(save_image_to_minio)  # Save image to MinIO
sdf = sdf.update(lambda value: print(f"✅ Processing complete") or value)

if __name__ == "__main__":
    app.run()