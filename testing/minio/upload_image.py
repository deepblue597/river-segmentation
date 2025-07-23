import boto3
from botocore.client import Config

s3 = boto3.resource('s3',
                    endpoint_url='http://linux-pc:9000',
                    aws_access_key_id='minio',
                    aws_secret_access_key='minio123',
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')  #



# Function to upload an image to MinIO
def upload_image(file_path, bucket_name, object_name=None):
    """
    Uploads an image to a specified MinIO bucket.

    :param file_path: Path to the image file to upload.
    :param bucket_name: Name of the MinIO bucket.
    :param object_name: Name of the object in the bucket. If not specified, file_path is used.
    """
    if object_name is None:
        object_name = file_path

    try:
        s3.Bucket(bucket_name).upload_file(file_path, object_name)
        print(f"Image {file_path} uploaded to {bucket_name}/{object_name}")
    except Exception as e:
        print(f"Failed to upload image: {e}")
        
        
# Example usage
if __name__ == "__main__":
    # Replace with your image file path and bucket name
    file_path = 'composite_0_13.png'
    bucket_name = 'river'
    
    upload_image(file_path, bucket_name)