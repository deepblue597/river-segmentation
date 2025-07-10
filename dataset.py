import os 
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class RiverDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        # Our directories for images and masks
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # list all files that are in the folder 
        self.img_names = os.listdir(img_dir)
        self.mask_names = os.listdir(mask_dir) 
        
        
    def __len__(self):
        # Return the number of images in the dataset
        return len(self.img_names)
    
    def __getitem__(self, idx):
        # Get the image and mask file names
        img_name = self.img_names[idx]
        mask_name = self.mask_names[idx]
        
        # Load the image and mask
        # In this case the image and mask have the same name and extension
        # but are in different directories
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
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