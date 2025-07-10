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