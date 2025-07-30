import json
from fastapi import FastAPI, Request
from river_segmentation.connectors import MinIOConnector
import time
from fastapi.responses import StreamingResponse
import io
from fastapi import File, UploadFile, HTTPException
import base64
import requests
from datetime import datetime
from river_segmentation.connectors import KafkaProducerConnector
import os
from response_models import *

app = FastAPI()

MINIO_ADDRESS = os.environ.get("MINIO_ADDRESS", "localhost")
MINIO_PORT = int(os.environ.get("MINIO_PORT", 9000))
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "river")

KAFKA_ADDRESS = os.environ.get("KAFKA_ADDRESS", "localhost")
KAFKA_PORT = int(os.environ.get("KAFKA_PORT", 39092))
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "River")
KAFKA_ACKS = os.environ.get("KAFKA_ACKS", "all")
KAFKA_COMPRESSION_TYPE = os.environ.get("KAFKA_COMPRESSION_TYPE", "snappy")


@app.get("/")
async def root():
    return {"message": "River Segmentation API is running", "status": "ok"}


minioClient = MinIOConnector(
    address=MINIO_ADDRESS,
    port=MINIO_PORT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    target=MINIO_BUCKET,
)

kafkaClient = KafkaProducerConnector(
    address=KAFKA_ADDRESS,
    port=KAFKA_PORT,
    topic=KAFKA_TOPIC,
    acks=KAFKA_ACKS,
    compression_type=KAFKA_COMPRESSION_TYPE,
)

minioClient.connect()
kafkaClient.connect()


@app.get("/get-location", response_model=LocationResponse)
async def get_location(request: Request):
    # Get client IP
    client_ip = request.client.host

    if client_ip in ["127.0.0.1", "::1", "localhost"]:
        print("⚠️ Localhost detected, using default location")
        return {
            "ip": client_ip,
            "country": "Development",
            "region": "Local",
            "city": "Localhost",
            "lat": 40.7128,  # Default to NYC coordinates
            "lon": -74.0060,
        }
    # Use an external API for geo lookup (e.g., ip-api.com, ipinfo.io, etc.)
    response = requests.get(f"http://ip-api.com/json/{client_ip}")

    if response.status_code == 200:
        data = response.json()
        res = LocationResponse(
            ip=client_ip,
            country=data.get("country"),
            region=data.get("regionName"),
            city=data.get("city"),
            lat=data.get("lat"),
            lon=data.get("lon"),
        )
        return res

    else:
        return {"error": "Could not get location"}


@app.get("/debug/list_objects")
async def list_objects():
    """Debug endpoint to list all objects in MinIO bucket"""
    try:
        # You'll need to add this method to your MinIOConnector class
        # For now, let's try using the s3_client directly
        response = minioClient.s3_client.list_objects_v2(Bucket=minioClient.target)
        objects = []
        if "Contents" in response:
            for obj in response["Contents"]:
                objects.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat(),
                    }
                )
        res = DebugListObjectsResponse(objects=objects, count=len(objects))
        return res
    except Exception as e:
        return {"error": f"Failed to list objects: {str(e)}"}


@app.get("/get_image/{object_name:path}")
async def get_image(object_name: str, timeout: int = 30):
    start = time.time()
    print(f"Fetching image: {object_name} with timeout: {timeout} seconds")

    while time.time() - start < timeout:
        try:
            print(f"Attempting to fetch object: {object_name}")
            data = minioClient.get_object(object_name)

            if data and len(data) > 0:
                print(
                    f"Successfully fetched image: {object_name}, size: {len(data)} bytes"
                )
                return StreamingResponse(io.BytesIO(data), media_type="image/png")
            else:
                print(f"Object {object_name} exists but is empty")

        except Exception as e:
            print(f"Error fetching {object_name}: {str(e)}")

        # Wait before trying again
        time.sleep(2)

    print(f"Timeout reached for {object_name}")
    raise HTTPException(
        status_code=404,
        detail=f"Image '{object_name}' not ready after {timeout} seconds",
    )


@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...), request: Request = None):
    try:
        # Read file content
        image_data = await file.read()

        # Upload to Kafka
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        location = await get_location(request)  # Get location data
        # print(image_base64)
        # Create message
        message = KafkaMessage(
            filename=file.filename,
            image=image_base64,
            date=datetime.now().isoformat(),
            file_size=len(image_data),
            lat=location.lat,  # Use lat from location
            lon=location.lon,  # Use lon from location
        )
        # message = {
        #     "filename": file.filename,
        #     "image": image_base64,
        #     "date": datetime.now().isoformat(),
        #     "file_size": len(image_data),
        #     "lat": location["lat"],  # Use city as upload source,
        #     "lon": location["lon"],
        # }

        kafkaClient.produce(
            key=str(datetime.now().timestamp()),  # Use timestamp as key
            value=json.dumps(message),
        )
        return {"status": "success", "message": "Image uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")
