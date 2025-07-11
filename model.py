import segmentation_models_pytorch as smp
import torch

# --- Step 6: Model, loss, optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b1",  # Better for water segmentation
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

model1 = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)