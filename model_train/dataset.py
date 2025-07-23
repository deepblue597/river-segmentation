import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class RiverDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        # Our file paths for images and masks
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
        # Ensure we have matching number of images and masks
        assert len(img_paths) == len(mask_paths), f"Mismatch: {len(img_paths)} images, {len(mask_paths)} masks" 
        
        
    def __len__(self):
        # Return the number of images in the dataset
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Get the image and mask file paths
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Open the image and mask files
        # RGB because we want to work with color images in River segmentation 
        # L because masks are binary (black and white)

        image = np.array(Image.open(img_path).convert("RGB")) 
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # we will convert the mask to float32 for processing
        # and normalize it to [0, 1] range
        # we use 255 as 1 because the value will either be 0 or 255 (2 colors only) as mentioned above 
        mask[mask == 255] = 1.0  # Convert mask to binary (0 and 1)
        #we will use sigmoid on our last activation layer
        
        if self.transform:
            # Apply any transformations if provided
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
            
        return image , mask