import json
from typing import Union
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from fastapi import FastAPI, Request
from connectors.minio_connector import MinIOConnector
import time
from fastapi.responses import StreamingResponse
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
import base64
import requests
from datetime import datetime

from kafka.producer import create_kafka_producer, delivery_callback

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "River Segmentation API is running", "status": "ok"}

minioClient = MinIOConnector(
    address='localhost',
    port=9000,
    access_key='minio',
    secret_key='minio123',
    target='river'
)


minioClient.connect()

@app.get("/get-location")
async def get_location(request: Request):
    # Get client IP
    client_ip = request.client.host
    
    if client_ip in ['127.0.0.1', '::1', 'localhost']:
        print("⚠️ Localhost detected, using default location")
        return {
            "ip": client_ip,
            "country": "Development",
            "region": "Local",
            "city": "Localhost",
            "lat": 40.7128,  # Default to NYC coordinates
            "lon": -74.0060
        }
    # Use an external API for geo lookup (e.g., ip-api.com, ipinfo.io, etc.)
    response = requests.get(f"http://ip-api.com/json/{client_ip}")
    
    if response.status_code == 200:
        data = response.json()
        return {
            "ip": client_ip,
            "country": data.get("country"),
            "region": data.get("regionName"),
            "city": data.get("city"),
            "lat": data.get("lat"),
            "lon": data.get("lon")
        }
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
        if 'Contents' in response:
            for obj in response['Contents']:
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                })
        return {"objects": objects, "count": len(objects)}
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
                print(f"Successfully fetched image: {object_name}, size: {len(data)} bytes")
                return StreamingResponse(io.BytesIO(data), media_type="image/png")
            else:
                print(f"Object {object_name} exists but is empty")
                
        except Exception as e:
            print(f"Error fetching {object_name}: {str(e)}")
            
        # Wait before trying again
        time.sleep(2)
    
    print(f"Timeout reached for {object_name}")
    raise HTTPException(status_code=404, detail=f"Image '{object_name}' not ready after {timeout} seconds")


@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...),  request: Request = None):
    
    try:
        # Read file content
        image_data = await file.read()
        
        # Upload to Kafka
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        location = await get_location(request) # Get location data
        # print(image_base64)
        # Create message
        message = {
            'filename': file.filename,
            'image': image_base64,
            'date': datetime.now().isoformat(),
            'file_size': len(image_data),
            'lat': location['lat'],  # Use city as upload source, 
            'lon': location['lon']
        }
        
        producer = create_kafka_producer(
                bootstrap_server='139.91.68.57:29092',
                acks='all',
                compression_type='snappy'
            )
        
        producer.produce(
            'River',
            value=json.dumps(message),
            key=str(datetime.now().timestamp()),
            callback=delivery_callback
        )
        producer.poll(0)
        producer.flush()
        return {"status": "success", "message": "Image uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")