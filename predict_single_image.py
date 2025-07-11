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
        A.Resize(height=384, width=384),  # Same as training
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
        keep_original_size: If True, resize outputs to original size. If False, resize original to 384x384
    
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
        original_image = cv2.resize(image, (384, 384))
        print("Processing at model resolution: 384x384")
    
    # Apply transformations
    # Applies the same preprocessing as training
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Ensure model is on the same device as input
    model.to(device)
    
    # Until now 
    
    # Original image: (1080, 1920, 3) - Height, Width, Channels
    # ↓
    # A.Resize(height=384, width=384)
    # Resized: (384, 384, 3)
    # ↓
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # Normalized: Each pixel value adjusted to match ImageNet statistics
    # ↓
    # ToTensorV2()
    # Tensor: (3, 384, 384) - Channels, Height, Width
    # ↓
    # .unsqueeze(0)
    # Batch tensor: (1, 3, 384, 384) - Batch, Channels, Height, Width
   
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
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
