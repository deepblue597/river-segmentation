"""
Simple script to test your trained water segmentation model on a single image
Usage: python predict_image.py
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model_train.model import model, device
from model_train.utils import load_checkpoint
import os

def predict_water_segmentation(image_path, model_path="best_model.pth.tar"):
    """
    Simple function to predict water segmentation on a single image
    
    Args:
        image_path: Path to your image
        model_path: Path to your trained model
    
    Returns:
        Shows visualization and saves results
    """
    
    # 1. Load the trained model
    print("Loading trained model...")
    if os.path.exists(model_path):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        load_checkpoint(model_path, model, optimizer)
        model.eval()
        print("✓ Model loaded successfully!")
    else:
        print(f"❌ Model file '{model_path}' not found!")
        return
    
    # 2. Load and preprocess the image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image_rgb.shape[:2]
    print(f"✓ Image loaded: {original_width}x{original_height}")
    
    # 3. Apply the same transformations as during training
    transform = A.Compose([
        A.Resize(height=384, width=384),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    # Transform the image
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # 4. Make prediction
    print("Making prediction...")
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction_probs = torch.sigmoid(prediction)
        confidence_map = prediction_probs.cpu().squeeze().numpy()
        prediction_mask = (confidence_map > 0.5).astype(np.uint8)
    
    # 5. Resize back to original dimensions
    prediction_mask_resized = cv2.resize(prediction_mask, (original_width, original_height))
    confidence_map_resized = cv2.resize(confidence_map, (original_width, original_height))
    
    # 6. Calculate statistics
    total_pixels = prediction_mask_resized.size
    water_pixels = np.sum(prediction_mask_resized)
    water_percentage = (water_pixels / total_pixels) * 100
    avg_confidence = np.mean(confidence_map_resized)
    
    print(f"✓ Prediction completed!")
    print(f"📊 Water coverage: {water_percentage:.1f}% of image")
    print(f"🎯 Average confidence: {avg_confidence:.3f}")
    
    # 7. Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Prediction mask
    axes[0, 1].imshow(prediction_mask_resized, cmap='gray')
    axes[0, 1].set_title('Water Mask', fontsize=14)
    axes[0, 1].axis('off')
    
    # Confidence map
    im = axes[1, 0].imshow(confidence_map_resized, cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title('Confidence Map', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = image_rgb.copy()
    # Create blue overlay for water areas
    overlay[prediction_mask_resized == 1] = [0, 100, 255]  # Blue
    # Blend with original
    blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
    
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title('Water Detection Overlay', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 8. Save results
    output_dir = "single_image_prediction"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mask
    cv2.imwrite(f"{output_dir}/water_mask.png", prediction_mask_resized * 255)
    
    # Save confidence map
    confidence_img = (confidence_map_resized * 255).astype(np.uint8)
    cv2.imwrite(f"{output_dir}/confidence_map.png", confidence_img)
    
    # Save overlay
    overlay_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/water_overlay.png", overlay_bgr)
    
    print(f"💾 Results saved to '{output_dir}/' folder")
    
    return {
        'water_percentage': water_percentage,
        'avg_confidence': avg_confidence,
        'prediction_mask': prediction_mask_resized,
        'confidence_map': confidence_map_resized
    }

if __name__ == "__main__":
    # Example usage - replace with your image path
    image_path = input("Enter path to your image: ")
    
    if os.path.exists(image_path):
        results = predict_water_segmentation(image_path)
    else:
        print("❌ Image file not found!")
        print("Make sure to provide the full path to your image file.")
        print("Example: C:/Users/username/Pictures/my_image.jpg")
