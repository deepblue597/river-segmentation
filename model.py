import segmentation_models_pytorch as smp
import torch

# --- Step 6: Model, loss, optimizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Different EfficientNet options - uncomment to try different models
# model = smp.UnetPlusPlus(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1)  # Fastest
# model = smp.UnetPlusPlus(encoder_name="efficientnet-b1", encoder_weights="imagenet", in_channels=3, classes=1)  # Balanced
# model = smp.UnetPlusPlus(encoder_name="efficientnet-b3", encoder_weights="imagenet", in_channels=3, classes=1)  # Higher accuracy
# model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights="imagenet", in_channels=3, classes=1)  # Best accuracy (if you have GPU memory)

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",  # Upgraded from b1 - better accuracy with your GPU memory
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    # Note: This UNet++ doesn't have built-in deep supervision
    # Deep supervision would require manual implementation
)

model1 = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)