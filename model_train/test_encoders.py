import segmentation_models_pytorch as smp

print("Testing UnetPlusPlus encoder compatibility:")
encoders = smp.encoders.get_encoder_names()
compatible = []
incompatible = []

for enc in sorted(encoders):
    try:
        # Test creating UnetPlusPlus model with this encoder
        model = smp.UnetPlusPlus(
            encoder_name=enc, encoder_weights=None, in_channels=3, classes=1
        )
        compatible.append(enc)
        print(f"✓ {enc}")
    except Exception as e:
        incompatible.append((enc, str(e)))
        print(f"✗ {enc}: {str(e)}")

print("\n=== SUMMARY ===")
print(f"Compatible encoders: {len(compatible)}")
print(f"Incompatible encoders: {len(incompatible)}")

print("\n=== RECOMMENDED ENCODERS ===")
# Highlight some popular and efficient encoders
recommended = [
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "se_resnet50",
    "se_resnet101",
    "resnext50_32x4d",
    "resnext101_32x4d",
    "mobilenet_v2",
]

print("Popular choices for segmentation:")
for enc in recommended:
    if enc in compatible:
        print(f"  ✓ {enc}")
