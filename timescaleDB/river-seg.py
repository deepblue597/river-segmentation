import psycopg2
from datetime import datetime
import time

def send_real_time_data(water_coverage, confidence, overflow_detected):
    """Send river segmentation results to TimescaleDB"""
    conn = psycopg2.connect(
        dbname="test",
        user="postgres", 
        password="password",
        host="linux-pc",
        port="5432"
    )
    
    with conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO river_inference 
            (timestamp, image_name, water_coverage_pct, avg_confidence, 
             overflow_detected, processing_time_ms, model_version, location)
            VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
        """, (
            f"river_img_{int(time.time())}.jpg",
            water_coverage,
            confidence,
            overflow_detected,
            250,  # processing time
            "efficientnet-b3",
            "site_A"
        ))
        conn.commit()

# Usage in your model prediction pipeline
send_real_time_data(75.5, 0.92, True)