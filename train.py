import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch 
from tqdm import tqdm
from model import model, device
from utils import save_checkpoint, load_checkpoint, check_accuracy , get_loaders,   save_predictions_as_imgs
import os

# Function to read a text file and return a list of full paths
def read_txt_to_list(txt_path, dataset_root):
    with open(txt_path, 'r') as f:
        folders = [line.strip() for line in f.readlines() if line.strip()]
    full_paths = [os.path.join(dataset_root, folder) for folder in folders]
    return full_paths

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = device 
BATCH_SIZE = 32
NUM_EPOCHS = 10
# num_workers=2 → load data with 2 parallel workers (adjust for your CPU)
NUM_WORKERS = 8 
# we can change that depending onr our GPU memory
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
# pin_memory=True → speeds GPU data transfer
# When set to True, the DataLoader will copy Tensors into pinned (page-locked) memory before transferring them to the GPU.
PIN_MEMORY = True

#         True: Load a previously saved model checkpoint before training or evaluation.
#        (e.g., to continue training from where you left off or to evaluate a pretrained model)

#        False: Start training from scratch (randomly initialized weights).
LOAD_MODEL = True

path = kagglehub.dataset_download("gvclsu/water-segmentation-dataset")

print("Path to dataset files:", path)
DATASET_ROOT_IMAGE = os.path.join(path,"water_v2", "water_v2", "JPEGImages")
DATASET_ROOT_MASK = os.path.join(path,"water_v2", "water_v2", "Annotations")
TRAIN_TXT_FILE = os.path.join(path,"water_v2", "water_v2", "train.txt")
VAL_TXT_FILE = os.path.join(path,"water_v2", "water_v2", "val.txt")

TRAIN_FOLDERS = read_txt_to_list(TRAIN_TXT_FILE, DATASET_ROOT_IMAGE)
VAL_FOLDERS = read_txt_to_list(VAL_TXT_FILE, DATASET_ROOT_MASK)


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

        with torch.cuda.amp.autocast():
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
            A.Rotate(limit=35, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=(0, 0, 0), 
                std=(1, 1, 1), 
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=(0, 0, 0), 
                std=(1, 1, 1), 
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    
    model.to(device=DEVICE)
    # loss function 
    loss = torch.nn.BCEWithLogitsLoss()

    #optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader , val_loader = get_loaders(
        train_folders=TRAIN_FOLDERS,
        val_folders=VAL_FOLDERS,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train the model
        train(train_loader, model, optimizer, loss, scaler)
       
         
        # Check accuracy on validation set
        check_accuracy(val_loader, model, device=DEVICE)
        
        # Save model checkpoint
        if LOAD_MODEL:
            save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar")
    
if __name__ == "__main__": 
    main() 