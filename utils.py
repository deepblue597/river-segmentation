import os
import torch
import torchvision
from dataset import RiverDataset
from torch.utils.data import DataLoader
import random
from sklearn.model_selection import train_test_split


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """Save model checkpoint"""
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer):
    """Load model checkpoint"""
    print("=> Loading checkpoint")
    # map_location="cpu" is a parameter that tells PyTorch where to load the checkpoint data. 
    # It's crucial for compatibility across different devices.
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def check_accuracy(loader, model, device="cuda"):
    """Check accuracy on validation/test set"""
    num_correct = 0
    num_pixels = 0
    
    #The Dice Score (also called Dice Coefficient or F1-Score for segmentation) is a metric 
    # that measures the overlap between predicted and actual segmentation masks. 
    # It's widely used in medical imaging and segmentation tasks.
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            #NOTE: i am not sure for the unsqueeze(1) here,
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            #Ground Truth is Binary
            #Your masks are binary (0 or 1):
            preds = (preds > 0.5).float()
            
            # num_correct accumulates the sum of correct pixels from ALL batches, not just the current batch.
            num_correct += (preds == y).sum()
            #Returns the total number of elements in a tensor.
            num_pixels += torch.numel(preds)
            # sum the pixels where they are the same then divide by the number of pixels that output one for both of them
            # Google it for more info 
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}%"
    )
    dice_avg = dice_score/len(loader)
    print(f"Dice score: {dice_avg:.4f}")
    model.train()
    
    return dice_avg  # Return dice score for tracking best model


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    """Save predictions as images"""
    model.eval()
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        # With gradients (training):
        # - Stores intermediate activations for backpropagation
        # - Memory usage: ~2x higher
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        # Save original input image
        torchvision.utils.save_image(
            x, f"{folder}/input_{idx}.png"
        )
        
        # Save prediction mask
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        
        # Save target mask  
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/target_{idx}.png")

    model.train()


def split_dataset(image_dir, mask_dir, train_ratio=0.7, val_ratio=0.16, test_ratio=0.14, random_seed=42):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        train_ratio: Ratio for training set (default 0.7 = 70%)
        val_ratio: Ratio for validation set (default 0.16 = 16%)
        test_ratio: Ratio for test set (default 0.14 = 14%)
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (train_images, train_masks, val_images, val_masks, test_images, test_masks)
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # Sort for consistency
    
    # Get corresponding mask files
    mask_files = []
    for img_file in image_files:
        # Assuming masks have same name as images (might need adjustment based on your dataset)
        mask_file = img_file
        mask_path = os.path.join(mask_dir, mask_file)
        if os.path.exists(mask_path):
            mask_files.append(mask_file)
        else:
            print(f"Warning: Mask not found for {img_file}")
    
    # Ensure we have matching image-mask pairs
    assert len(image_files) == len(mask_files), f"Mismatch: {len(image_files)} images, {len(mask_files)} masks"
    
    # Create pairs
    data_pairs = list(zip(image_files, mask_files))
    
    # First split: train vs (val + test)
    train_pairs, temp_pairs = train_test_split(
        data_pairs, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_seed
    )
    
    # Second split: val vs test
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=(test_ratio / (val_ratio + test_ratio)),  # Adjust ratio for remaining data
        random_state=random_seed
    )
    
    # Extract file paths
    train_images = [os.path.join(image_dir, pair[0]) for pair in train_pairs]
    train_masks = [os.path.join(mask_dir, pair[1]) for pair in train_pairs]
    
    val_images = [os.path.join(image_dir, pair[0]) for pair in val_pairs]
    val_masks = [os.path.join(mask_dir, pair[1]) for pair in val_pairs]
    
    test_images = [os.path.join(image_dir, pair[0]) for pair in test_pairs]
    test_masks = [os.path.join(mask_dir, pair[1]) for pair in test_pairs]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_images)} images ({len(train_images)/len(data_pairs)*100:.1f}%)")
    print(f"  Val:   {len(val_images)} images ({len(val_images)/len(data_pairs)*100:.1f}%)")
    print(f"  Test:  {len(test_images)} images ({len(test_images)/len(data_pairs)*100:.1f}%)")
    
    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def get_loaders(
    train_images,
    train_masks,
    val_images,
    val_masks,
    train_transform,
    val_transform,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    """
    Create DataLoaders for training and validation
    
    Args:
        train_images: List of training image paths
        train_masks: List of training mask paths
        val_images: List of validation image paths
        val_masks: List of validation mask paths
        train_transform: Transformations for training
        val_transform: Transformations for validation
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to use pinned memory
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    train_ds = RiverDataset(
        img_paths = train_images,
        mask_paths = train_masks,
        transform=train_transform,
    )

    #DataLoader is a PyTorch class that provides an efficient and flexible way to load data for training and inference. 
    #It's a crucial component that handles batching, shuffling, and parallel data loading.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = RiverDataset(
        val_images,
        val_masks,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_test_loader(
    test_images,
    test_masks,
    test_transform,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    """
    Create DataLoader for testing
    
    Args:
        test_images: List of test image paths
        test_masks: List of test mask paths
        test_transform: Transformations for testing
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to use pinned memory
    
    Returns:
        DataLoader: test_loader
    """
    test_ds = RiverDataset(
        test_images,
        test_masks,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return test_loader
