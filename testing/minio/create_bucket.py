import boto3
from botocore.exceptions import ClientError

# MinIO configuration
minio_endpoint = "http://100.85.136.28:9000"
access_key = "minio"
secret_key = "minio123"
bucket_name = "river"

# Create S3 client for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url=minio_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name='eu-west-2'
)

try:
    # Create bucket
    s3_client.create_bucket(Bucket=bucket_name)
    print(f"✅ Bucket '{bucket_name}' created successfully!")
except ClientError as e:
    if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
        print(f"✅ Bucket '{bucket_name}' already exists!")
    else:
        print(f"❌ Error creating bucket: {e}")

# List buckets to verify
try:
    buckets = s3_client.list_buckets()
    print("\n📁 Available buckets:")
    for bucket in buckets['Buckets']:
        print(f"  - {bucket['Name']}")
except ClientError as e:
    print(f"❌ Error listing buckets: {e}")
