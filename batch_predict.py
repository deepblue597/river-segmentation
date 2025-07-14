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

def batch_predict_water_segmentation(input_folder, output_folder="batch_predictions", model_path="best_model.pth.tar", mask_folder=None):
    """
    Process multiple images in a folder
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder to save results
        model_path: Path to trained model
        mask_folder: Optional folder containing ground truth masks (same names as images)
    """
    
    # Use local device variable to avoid conflicts
    current_device = device
    
    # Load model
    print("Loading trained model...")
    print(f"Using device: {current_device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check if CUDA is available when device is set to cuda
    if current_device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        current_device = "cpu"
    
    if os.path.exists(model_path):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        load_checkpoint(model_path, model, optimizer)
        
        # Ensure model is on the correct device AFTER loading checkpoint
        model.to(current_device)
        model.eval()
        print(f"✓ Model loaded successfully on device: {current_device}")
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
            
            # Check device compatibility and fallback to CPU if needed
            try:
                input_tensor = transformed['image'].unsqueeze(0).to(current_device)
                
                # Ensure model is on the same device as input
                model.to(current_device)
                
                with torch.no_grad():
                    prediction = model(input_tensor)
                    prediction_probs = torch.sigmoid(prediction)
                    confidence_map = prediction_probs.cpu().squeeze().numpy()
                    prediction_mask = (confidence_map > 0.5).astype(np.uint8)
                    
            except RuntimeError as e:
                if "cuda" in str(e).lower():
                    print(f"⚠️ CUDA error for {base_name}, falling back to CPU")
                    # Fallback to CPU
                    input_tensor = transformed['image'].unsqueeze(0).to('cpu')
                    model.to('cpu')
                    
                    with torch.no_grad():
                        prediction = model(input_tensor)
                        prediction_probs = torch.sigmoid(prediction)
                        confidence_map = prediction_probs.cpu().squeeze().numpy()
                        prediction_mask = (confidence_map > 0.5).astype(np.uint8)
                    
                    # Move model back to original device for next iteration
                    model.to(current_device)
                else:
                    raise e
            
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
            
            # Load ground truth mask if available
            true_mask = None
            if mask_folder and os.path.exists(mask_folder):
                # Try different mask file extensions
                mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
                for ext in mask_extensions:
                    mask_path = os.path.join(mask_folder, base_name + ext)
                    if os.path.exists(mask_path):
                        true_mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if true_mask_img is not None:
                            # Resize to match prediction size
                            true_mask = cv2.resize(true_mask_img, (original_width, original_height))
                            # Convert to binary (0 or 1)
                            true_mask = (true_mask > 127).astype(np.uint8)
                            break
            
            # Save mask
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), 
                       prediction_mask_resized * 255)
            
            # Save confidence map
            confidence_img = (confidence_map_resized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_confidence.png"), 
                       confidence_img)
            
            # Create overlay with comparison if true mask is available
            if true_mask is not None:
                # Create composite overlay with both prediction and ground truth
                overlay = np.zeros_like(image_rgb)
                
                # Blue overlay for predictions
                overlay[prediction_mask_resized == 1] = [0, 0, 255]  # Blue for predicted water
                
                # Red overlay for true mask
                overlay[true_mask == 1] = [255, 0, 0]  # Red for true water
                
                # Green overlay where both prediction and true mask overlap
                overlap_mask = (prediction_mask_resized == 1) & (true_mask == 1)
                overlay[overlap_mask] = [0, 255, 0]  # Green for correct predictions
                
                # Blend original image with overlays
                alpha = 0.4
                composite = cv2.addWeighted(image_rgb, 1-alpha, overlay, alpha, 0)
                
                # Save composite overlay
                composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_folder, f"{base_name}_comparison.png"), 
                           composite_bgr)
                
                # Calculate performance metrics
                intersection = np.sum(overlap_mask)
                union = np.sum((prediction_mask_resized == 1) | (true_mask == 1))
                dice_score = (2 * intersection) / (np.sum(prediction_mask_resized) + np.sum(true_mask) + 1e-8)
                iou_score = intersection / (union + 1e-8)
                
                results.append({
                    'image': base_name,
                    'water_percentage': water_percentage,
                    'avg_confidence': avg_confidence,
                    'dice_score': dice_score,
                    'iou_score': iou_score,
                    'has_ground_truth': True
                })
            else:
                # Save regular overlay (prediction only)
                overlay = image_rgb.copy()
                overlay[prediction_mask_resized == 1] = [0, 100, 255]  # Blue
                blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
                overlay_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_folder, f"{base_name}_overlay.png"), 
                           overlay_bgr)
                
                results.append({
                    'image': base_name,
                    'water_percentage': water_percentage,
                    'avg_confidence': avg_confidence,
                    'has_ground_truth': False
                })
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            continue
    
    # Save summary
    summary_path = os.path.join(output_folder, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Water Segmentation Results Summary\n")
        f.write("=" * 40 + "\n\n")
        
        # Calculate overall statistics
        total_images = len(results)
        images_with_gt = sum(1 for r in results if r['has_ground_truth'])
        
        if images_with_gt > 0:
            avg_dice = np.mean([r['dice_score'] for r in results if r['has_ground_truth']])
            avg_iou = np.mean([r['iou_score'] for r in results if r['has_ground_truth']])
            f.write(f"Overall Performance (on {images_with_gt} images with ground truth):\n")
            f.write(f"Average Dice Score: {avg_dice:.3f}\n")
            f.write(f"Average IoU Score: {avg_iou:.3f}\n")
            f.write("\n" + "=" * 40 + "\n\n")
        
        for result in results:
            f.write(f"Image: {result['image']}\n")
            f.write(f"Water Coverage: {result['water_percentage']:.1f}%\n")
            f.write(f"Avg Confidence: {result['avg_confidence']:.3f}\n")
            if result['has_ground_truth']:
                f.write(f"Dice Score: {result['dice_score']:.3f}\n")
                f.write(f"IoU Score: {result['iou_score']:.3f}\n")
            f.write("-" * 30 + "\n")
    
    print(f"✓ Processed {len(results)} images successfully")
    if any(r['has_ground_truth'] for r in results):
        print(f"🎯 Performance metrics calculated for images with ground truth")
        print(f"📊 Comparison overlays saved as '*_comparison.png'")
    print(f"💾 Results saved to '{output_folder}/' folder")
    print(f"📊 Summary saved to '{summary_path}'")

if __name__ == "__main__":
    # Example usage
    input_folder = input("Enter path to folder containing images: ")
    
    if os.path.exists(input_folder):
        # Ask if user has ground truth masks
        has_masks = input("Do you have ground truth masks? (y/n): ").lower().strip()
        
        if has_masks == 'y':
            mask_folder = input("Enter path to folder containing masks: ")
            if os.path.exists(mask_folder):
                batch_predict_water_segmentation(input_folder, mask_folder=mask_folder)
            else:
                print("❌ Mask folder not found! Running without ground truth comparison.")
                batch_predict_water_segmentation(input_folder)
        else:
            batch_predict_water_segmentation(input_folder)
    else:
        print("❌ Folder not found!")
        print("Make sure to provide the full path to your image folder.")
        print("Example: C:/Users/username/Pictures/my_images/")
