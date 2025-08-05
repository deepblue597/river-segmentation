from datetime import datetime
import random


# Example function to generate a single mock row
def generate_mock_row():
    timestamp = datetime.now()
    image_name = f"river_image_{random.randint(1000, 9999)}.jpg"
    water_coverage_pct = round(random.uniform(10, 90), 2)
    avg_confidence = round(random.uniform(0.5, 1.0), 2)
    iou_score = round(random.uniform(0.3, 0.9), 2)
    dice_score = round(random.uniform(0.3, 0.9), 2)
    overflow_detected = water_coverage_pct > 60
    processing_time_ms = random.randint(50, 500)
    model_version = "v1.0"
    location = random.choice(["RiverA", "RiverB", "RiverC"])

    return {
        "timestamp": timestamp,
        "image_name": image_name,
        "water_coverage_pct": water_coverage_pct,
        "avg_confidence": avg_confidence,
        "iou_score": iou_score,
        "dice_score": dice_score,
        "overflow_detected": overflow_detected,
        "processing_time_ms": processing_time_ms,
        "model_version": model_version,
        "location": location,
    }
