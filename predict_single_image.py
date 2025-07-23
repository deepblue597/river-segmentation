from datetime import datetime
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from model import model, device
from utils import load_checkpoint
import boto3
from sqlalchemy import create_engine , Column , Integer , String , Float , Boolean , MetaData , Table, text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy import insert , select
import base64

# Load model once when module is imported
GLOBAL_MODEL = None
GLOBAL_TRANSFORM = None

def initialize_model():
    """Initialize the global model and transform - call this once at startup"""
    global GLOBAL_MODEL, GLOBAL_TRANSFORM
    if GLOBAL_MODEL is None:
        print("Initializing model for the first time...")
        GLOBAL_MODEL = load_trained_model("best_model.pth.tar")
        GLOBAL_TRANSFORM = get_prediction_transform()
        print("Model initialization complete!")
    return GLOBAL_MODEL is not None

def load_trained_model(checkpoint_path="best_model.pth.tar"):
    """Load the trained model from checkpoint"""
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        # Create a dummy optimizer for loading checkpoint
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Load checkpoint
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Move model to the correct device
        model.to(device)
        model.eval()
        return model
    else:
        print(f"Checkpoint {checkpoint_path} not found!")
        return None

def get_prediction_transform():
    """Get transformation for single image prediction"""
    return A.Compose([
        A.Resize(height=512, width=512),  # Same as training
        A.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet mean
            std=(0.229, 0.224, 0.225),   # ImageNet std
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

def predict_single_image(image_path, model, transform, device="cuda", keep_original_size=True):
    """
    Predict water segmentation for a single image
    
    Args:
        image_path: Path to the input image
        model: Trained model
        transform: Image transformation
        device: Device to run inference on
        keep_original_size: If True, resize outputs to original size. If False, resize original to 512x512
    
    Returns:
        original_image: Original image as numpy array
        prediction_mask: Predicted mask as numpy array
        confidence_map: Confidence map before thresholding
    """
    # Check if CUDA is available, otherwise use CPU
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    # BGR → RGB: OpenCV uses BGR, but most ML models expect RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if keep_original_size:
        # Approach 1: Keep original size, resize predictions back
        original_image = image.copy()
        original_height, original_width = original_image.shape[:2]
        print(f"Processing at original resolution: {original_width}x{original_height}")
    else:
        # Approach 2: Resize original to match prediction size
        original_image = cv2.resize(image, (512, 512))
        print("Processing at model resolution: 512x512")
    
    # Apply transformations
    # Applies the same preprocessing as training
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Ensure model is on the same device as input
    model.to(device)
    
    # Until now 
    
    # Original image: (1080, 1920, 3) - Height, Width, Channels
    # ↓
    # A.Resize(height=512, width=512)
    # Resized: (512, 512, 3)
    # ↓
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # Normalized: Each pixel value adjusted to match ImageNet statistics
    # ↓
    # ToTensorV2()
    # Tensor: (3, 512, 512) - Channels, Height, Width
    # ↓
    # .unsqueeze(0)
    # Batch tensor: (1, 3, 512, 512) - Batch, Channels, Height, Width
   
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        # Apply sigmoid to get probabilities
        prediction_probs = torch.sigmoid(prediction)
        # Convert to numpy
        confidence_map = prediction_probs.cpu().squeeze().numpy()
        # Apply threshold
        prediction_mask = (confidence_map > 0.5).astype(np.uint8)
    
    if keep_original_size:
        # Resize outputs back to original image dimensions
        confidence_map_resized = cv2.resize(confidence_map, (original_width, original_height))
        prediction_mask_resized = cv2.resize(prediction_mask, (original_width, original_height))
        
        # Ensure binary mask stays binary after resize
        prediction_mask_resized = (prediction_mask_resized > 0.5).astype(np.uint8)
        
        return original_image, prediction_mask_resized, confidence_map_resized
    else:
        # Return everything at 384x384 resolution
        return original_image, prediction_mask, confidence_map

def visualize_prediction(original_image, prediction_mask, confidence_map, save_path=None):
    """
    Visualize the prediction results
    
    Args:
        original_image: Original image
        prediction_mask: Binary prediction mask
        confidence_map: Confidence map (0-1)
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Prediction mask
    axes[0, 1].imshow(prediction_mask, cmap='gray')
    axes[0, 1].set_title('Predicted Water Mask', fontsize=14)
    axes[0, 1].axis('off')
    
    # Confidence map
    im = axes[1, 0].imshow(confidence_map, cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title('Confidence Map', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = original_image.copy()
    # Create colored mask (blue for water)
    colored_mask = np.zeros_like(original_image)
    colored_mask[prediction_mask == 1] = [0, 0, 255]  # Blue for water
    
    # Blend original and mask
    alpha = 0.4
    overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Water Detection Overlay', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def save_prediction_masks(original_image, prediction_mask, confidence_map, output_dir="prediction_output"):
    """Save prediction results as separate files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image
    cv2.imwrite(os.path.join(output_dir, "original.png"), 
                cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    
    # Save binary mask
    cv2.imwrite(os.path.join(output_dir, "mask.png"), 
                prediction_mask * 255)
    
    # Save confidence map
    confidence_img = (confidence_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "confidence.png"), confidence_img)
    
    print(f"Prediction files saved to {output_dir}/")
    

def get_s3_client():
    """Get S3 client for saving results"""
    return  boto3.client(
        's3',
        endpoint_url='http://linux-pc:9000',
        aws_access_key_id='minio',
        aws_secret_access_key='minio123',
        region_name='eu-west-2'
    )

def save_mask_to_minio( prediction_mask , kafka_message):
    
    try: 
        image_name = kafka_message['filename']  #os.path.basename(original_image)
        encoded_mask = cv2.imencode('.png', prediction_mask * 255)[1].tobytes()
        
        s3_client = get_s3_client()
        
        s3_client.put_object(
            Bucket='river',
            Key=f"predictions/{image_name}",
            Body = encoded_mask, 
            ContentType='image/png'
        )
        
        #print(f"Mask saved to MinIO: predictions/{image_name}")
        return kafka_message  # Return the original message for further processing
        
    except Exception as e:
        print(f"Error saving mask to MinIO:")
        return None
        
def save_confidence_to_minio(original_image, confidence_map, kafka_message):
    try:
        image_name = kafka_message['filename']  #os.path.basename(original_image)
        encoded_confidence = cv2.imencode('.png', (confidence_map * 255).astype(np.uint8))[1].tobytes()
        
        s3_client = get_s3_client()
        
        s3_client.put_object(
            Bucket='river',
            Key=f"confidence/{image_name}",
            Body=encoded_confidence,
            ContentType='image/png'
        )
        
        print(f"Confidence map saved to MinIO: predictions/confidence_{image_name}")
        return kafka_message  # Return the original message for further processing
        
    except Exception as e:
        print(f"Error saving confidence map to MinIO:")
        return None

def save_overlay_to_minio(original_image, prediction_mask, kafka_message):
    try:
        image_name = kafka_message['filename']  #os.path.basename(original_image)
        
        # Create overlay
        overlay = original_image.copy()
        overlay[prediction_mask == 1] = [0, 255, 0]  # Blue for water
        blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
        
        encoded_overlay = cv2.imencode('.png', blended)[1].tobytes()
        
        s3_client = get_s3_client()
        
        s3_client.put_object(
            Bucket='river',
            Key=f"overlays/{image_name}",
            Body=encoded_overlay,
            ContentType='image/png'
        )
        
        print(f"Overlay saved to MinIO: overlays/{image_name}")
        #return kafka_message  # Return the original message for further processing
        
    except Exception as e:
        print(f"Error saving overlay to MinIO:")
        return None

def save_statistics_to_timescale(water_percentage, avg_confidence, kafka_message):
    try: 
        print(f"Attempting to save statistics for image: {kafka_message.get('filename', 'unknown')}")
        
        # Create engine with more explicit error handling
        engine = create_engine(
            'postgresql+psycopg2://postgres:password@linux-pc:5432/river',
            echo=False  # Set to True for SQL debugging
        )
        
        # Test connection first
        with engine.connect() as test_conn:
            result = test_conn.execute(text("SELECT 1"))
            print("Database connection successful")
        
        metadata = MetaData()
        
        # Try to load the table
        try:
            river_segmentation_results = Table(
                'river_segmentation_results',
                metadata,
                autoload_with=engine
            )
            print("Table loaded successfully")
        except Exception as table_error:
            print(f"Error loading table: {table_error}")
            print("Available tables:")
            with engine.connect() as conn:
                tables_result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
                for row in tables_result:
                    print(f"  - {row[0]}")
            return None
        
        # Prepare data with validation
        filename = kafka_message.get('filename', 'unknown')
        processing_time = kafka_message.get('processing_time_ms', 0)
        model_version = kafka_message.get('model_version', 'unet++')
        location = kafka_message.get('location', 'unknown')
        
        data = {
            'timestamp': datetime.now(),
            'image_name': filename,
            'water_coverage_pct': float(water_percentage),
            'avg_confidence': float(avg_confidence),
            'iou_score': 0.0,  # Placeholder, calculate if needed
            'dice_score': 0.0,  # Placeholder, calculate if needed
            'overflow_detected': False,  # Placeholder, set based on your logic
            'processing_time_ms': int(processing_time),
            'model_version': model_version,
            'location': location
        }
        
        print(f"Data to insert: {data}")
        
        # Insert data
        with engine.connect() as conn:
            insert_stmt = insert(river_segmentation_results).values(data)
            result = conn.execute(insert_stmt)
            conn.commit()
            print(f"Statistics saved to TimescaleDB for image: {filename}")
            return True
    
    except Exception as e:
        print(f"Error saving statistics to TimescaleDB: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def predict_and_save(X):
    global GLOBAL_MODEL, GLOBAL_TRANSFORM
    
    # Initialize model if not already done
    if not initialize_model():
        print("Failed to initialize model")
        return
    
    # Print device information (only first time)
    if GLOBAL_MODEL is not None:
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Use the global model and transform
    trained_model = GLOBAL_MODEL
    transform = GLOBAL_TRANSFORM
    
    image_data = X['image']
    image_bytes = base64.b64decode(image_data)
    
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Could not decode image from bytes")
        return
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Since we're using keep_original_size=False, resize to model resolution
    original_image = cv2.resize(image, (512, 512))
    print("Processing at model resolution: 512x512")
    
    # Apply transformations
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Make prediction
    try:
        with torch.no_grad():
            prediction = trained_model(input_tensor)
            # Apply sigmoid to get probabilities
            prediction_probs = torch.sigmoid(prediction)
            # Convert to numpy
            confidence_map = prediction_probs.cpu().squeeze().numpy()
            # Apply threshold
            prediction_mask = (confidence_map > 0.5).astype(np.uint8)
        
        # Calculate statistics
        total_pixels = prediction_mask.size
        water_pixels = np.sum(prediction_mask)
        water_percentage = (water_pixels / total_pixels) * 100
        avg_confidence = np.mean(confidence_map)
        
        print(f"Prediction completed!")
        print(f"Water pixels: {water_pixels:,} / {total_pixels:,} ({water_percentage:.1f}%)")
        print(f"Average confidence: {avg_confidence:.3f}")
    
        # Save to MinIO
        save_mask_to_minio(prediction_mask, X)
        save_confidence_to_minio(original_image, confidence_map, X)
        save_overlay_to_minio(original_image, prediction_mask, X)
        # Save statistics to TimescaleDB
        save_statistics_to_timescale(water_percentage, avg_confidence, X)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        #import traceback
        #traceback.print_exc()
    
def main():
    parser = argparse.ArgumentParser(description='Predict water segmentation on a single image')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='best_model.pth.tar',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='prediction_output',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization')
    parser.add_argument('--save-viz', type=str, default=None,
                        help='Save visualization to file')
    parser.add_argument('--low-res', action='store_true',
                        help='Process at 384x384 resolution (faster, lower quality)')
    
    args = parser.parse_args()
    
    # Print device information
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load trained model
    trained_model = load_trained_model(args.model)
    if trained_model is None:
        return
    
    # Get transformation
    transform = get_prediction_transform()
    
    # Make prediction
    try:
        original_image, prediction_mask, confidence_map = predict_single_image(
            args.image, trained_model, transform, device, 
            keep_original_size=not args.low_res  # Use low_res flag to control behavior
        )
        
        # Calculate statistics
        total_pixels = prediction_mask.size
        water_pixels = np.sum(prediction_mask)
        water_percentage = (water_pixels / total_pixels) * 100
        avg_confidence = np.mean(confidence_map)
        
        print(f"Prediction completed!")
        print(f"Water pixels: {water_pixels:,} / {total_pixels:,} ({water_percentage:.1f}%)")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Save results
        save_prediction_masks(original_image, prediction_mask, confidence_map, args.output)
        
        # Visualize if requested
        if args.visualize:
            # Auto-generate filename if save path not provided
            if args.save_viz is None:
                base_name = os.path.splitext(os.path.basename(args.image))[0]
                timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                args.save_viz = f"{args.output}/visualization_{base_name}_{timestamp}.png"
            
            visualize_prediction(original_image, prediction_mask, confidence_map, args.save_viz)
        
        # Save to MinIO
        save_mask_to_minio(original_image, prediction_mask, kafka_message)
        save_confidence_to_minio(original_image, confidence_map, kafka_message)
        save_overlay_to_minio(original_image, prediction_mask, kafka_message)
        # Save statistics to TimescaleDB
        save_statistics_to_timescale(water_percentage, avg_confidence, kafka_message)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    