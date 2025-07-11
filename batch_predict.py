"""
Batch processing script for water segmentation on multiple images
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
import glob
from model import model, device
from utils import load_checkpoint
from tqdm import tqdm

def batch_predict_water_segmentation(input_folder, output_folder="batch_predictions", model_path="best_model.pth.tar"):
    """
    Process multiple images in a folder
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder to save results
        model_path: Path to trained model
    """
    
    # Load model
    print("Loading trained model...")
    if os.path.exists(model_path):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        load_checkpoint(model_path, model, optimizer)
        model.eval()
        print("✓ Model loaded successfully!")
    else:
        print(f"❌ Model file '{model_path}' not found!")
        return
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        print(f"❌ No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Setup transformation
    transform = A.Compose([
        A.Resize(height=384, width=384),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    # Process each image
    results = []
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load {image_path}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image_rgb.shape[:2]
            
            # Transform and predict
            transformed = transform(image=image_rgb)
            input_tensor = transformed['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction = model(input_tensor)
                prediction_probs = torch.sigmoid(prediction)
                confidence_map = prediction_probs.cpu().squeeze().numpy()
                prediction_mask = (confidence_map > 0.5).astype(np.uint8)
            
            # Resize back to original dimensions
            prediction_mask_resized = cv2.resize(prediction_mask, (original_width, original_height))
            confidence_map_resized = cv2.resize(confidence_map, (original_width, original_height))
            
            # Calculate statistics
            total_pixels = prediction_mask_resized.size
            water_pixels = np.sum(prediction_mask_resized)
            water_percentage = (water_pixels / total_pixels) * 100
            avg_confidence = np.mean(confidence_map_resized)
            
            # Save results
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save mask
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), 
                       prediction_mask_resized * 255)
            
            # Save confidence map
            confidence_img = (confidence_map_resized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_confidence.png"), 
                       confidence_img)
            
            # Save overlay
            overlay = image_rgb.copy()
            overlay[prediction_mask_resized == 1] = [0, 100, 255]  # Blue
            blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
            overlay_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_overlay.png"), 
                       overlay_bgr)
            
            results.append({
                'image': base_name,
                'water_percentage': water_percentage,
                'avg_confidence': avg_confidence
            })
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue
    
    # Save summary
    summary_path = os.path.join(output_folder, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Water Segmentation Results Summary\n")
        f.write("=" * 40 + "\n\n")
        for result in results:
            f.write(f"Image: {result['image']}\n")
            f.write(f"Water Coverage: {result['water_percentage']:.1f}%\n")
            f.write(f"Avg Confidence: {result['avg_confidence']:.3f}\n")
            f.write("-" * 30 + "\n")
    
    print(f"✓ Processed {len(results)} images successfully")
    print(f"💾 Results saved to '{output_folder}/' folder")
    print(f"📊 Summary saved to '{summary_path}'")

if __name__ == "__main__":
    # Example usage
    input_folder = input("Enter path to folder containing images: ")
    
    if os.path.exists(input_folder):
        batch_predict_water_segmentation(input_folder)
    else:
        print("❌ Folder not found!")
        print("Make sure to provide the full path to your image folder.")
        print("Example: C:/Users/username/Pictures/my_images/")
