import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch 
from tqdm import tqdm
from model import model, device
from utils import save_checkpoint, load_checkpoint, check_accuracy, get_loaders, get_test_loader, save_predictions_as_imgs, split_dataset
import os
import kagglehub

# Hyperparameters - Optimized for Water Segmentation
LEARNING_RATE = 3e-4  # Good for larger batch sizes
DEVICE = device 
BATCH_SIZE = 32       # Increased from 16 - utilize your 40GB GPU!
NUM_EPOCHS = 100      # Increased from 50
NUM_WORKERS = 8       # Increased from 4 to utilize CPU cores better
IMAGE_HEIGHT = 512    # Increased from 384 for better detail capture
IMAGE_WIDTH = 512     # Higher resolution for better water boundary detection
PIN_MEMORY = True

# Additional optimizations for 40GB GPU
# ACCUMULATE_GRAD_BATCHES = 1  # Can increase if you want even larger effective batch size
# PREFETCH_FACTOR = 2          # Prefetch more batches for faster data loading

#         True: Load a previously saved model checkpoint before training or evaluation.
#        (e.g., to continue training from where you left off or to evaluate a pretrained model)

#        False: Start training from scratch (randomly initialized weights).
LOAD_MODEL = False

path = kagglehub.dataset_download("gvclsu/water-segmentation-dataset")

print("Path to dataset files:", path)
DATASET_ROOT_IMAGE = os.path.join(path,"water_v2", "water_v2", "JPEGImages" , "ADE20K")
DATASET_ROOT_MASK = os.path.join(path,"water_v2", "water_v2", "Annotations", "ADE20K")

# Split dataset into train (70%), validation (16%), and test (14%)
train_images, train_masks, val_images, val_masks, test_images, test_masks = split_dataset(
    image_dir=DATASET_ROOT_IMAGE,
    mask_dir=DATASET_ROOT_MASK,
    train_ratio=0.7,
    val_ratio=0.16,
    test_ratio=0.14,
    random_seed=42
)


"""
    1 epoch of training 
"""
def train(loader , model , optimizer , loss , scaler) : 
    loop = tqdm(loader) 
    
    for batch_idx, (data, targets) in enumerate(loop):
        
        data = data.to(device=DEVICE)
        # it is already in float 
        #Your dataset returns 2D masks [H, W]
        #After batching, they become [N, H, W]
        #Most segmentation loss functions (like BCEWithLogitsLoss, CrossEntropyLoss) expect [N, C, H, W] format
        #The unsqueeze(1) adds the channel dimension: [N, H, W] → [N, 1, H, W]
        targets = targets.unsqueeze(1).to(device=DEVICE)

        with torch.amp.autocast('cuda'):
            # Forward pass
            predictions = model(data)
            # Calculate loss
            loss_value = loss(predictions, targets)
        
        # Backward pass and optimization
        # Clears (zeros out) all gradients from the previous iteration before computing new gradients.
        optimizer.zero_grad()
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop 
        loop.set_postfix(loss=loss_value.item())
        
        
        

def main():
    """
Original: RGB image (1920x1080) + Binary mask (1920x1080)
↓
Resize: RGB image (512x512) + Binary mask (512x512)
↓
Rotate: Both rotated by +20° (same angle)
↓
H.Flip: Both flipped horizontally (or not, 50% chance)
↓
V.Flip: Both flipped vertically (or not, 10% chance)
↓
Normalize: Image pixels [0,255] → [0,1], mask unchanged
↓
ToTensor: NumPy arrays → PyTorch tensors, HWC → CHW
    
    """
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=20, p=0.7),  # Less aggressive rotation
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            # Water-specific augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet mean
                std=(0.229, 0.224, 0.225),   # ImageNet std
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet mean
                std=(0.229, 0.224, 0.225),   # ImageNet std
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    
    model.to(device=DEVICE)
    # loss function 
    loss = torch.nn.BCEWithLogitsLoss()

    #optimizer - AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=False
    )
    
    train_loader, val_loader = get_loaders(
        train_images=train_images,
        train_masks=train_masks,
        val_images=val_images,
        val_masks=val_masks,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # Advanced optimizations for large GPU memory
    scaler = torch.amp.GradScaler()  # For mixed precision training
    
    # Track best model performance
    best_dice_score = 0.0
    
    # Load checkpoint if specified
    if LOAD_MODEL and os.path.exists("my_checkpoint.pth.tar"):
        load_checkpoint("my_checkpoint.pth.tar", model, optimizer)
        print("Loaded old checkpoint format")
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train the model
        train(train_loader, model, optimizer, loss, scaler)
       
        # Check accuracy on validation set and get dice score
        current_dice = check_accuracy(val_loader, model, device=DEVICE)
        
        # Update learning rate based on validation performance
        scheduler.step(current_dice)
        
        # Print current learning rate
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")
        
        # Save checkpoint every epoch (for resuming training)
        save_checkpoint(model, optimizer, filename="latest_checkpoint.pth.tar")
        
        # Save best model if current performance is better
        if current_dice > best_dice_score:
            best_dice_score = current_dice
            save_checkpoint(model, optimizer, filename="best_model.pth.tar")
            print(f"New best model saved! Dice score: {best_dice_score:.4f}")

        #Great question! There are several reasons why save_predictions_as_imgs is not typically included in the training loop:
        # If we added this here:
        # save_predictions_as_imgs(val_loader, model, folder=f"epoch_{epoch}/")
        # This would save images 10 times (once per epoch)
        # Each save takes time and disk space
    # Final evaluation on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    # Load best model for final evaluation
    if os.path.exists("best_model.pth.tar"):
        load_checkpoint("best_model.pth.tar", model, optimizer)
        print("Loaded best model for final evaluation")
    else:
        print("No best model found, using current model")
    
    # Create test loader
    test_loader = get_test_loader(
        test_images=test_images,
        test_masks=test_masks,
        test_transform=val_transform,  # Use same transform as validation
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # Test the model
    check_accuracy(test_loader, model, device=DEVICE)
    
    # Save some test predictions
    save_predictions_as_imgs(
        test_loader, model, folder="test_predictions_efficientnet-b4/", device=DEVICE
    )
    
if __name__ == "__main__": 
    main() 