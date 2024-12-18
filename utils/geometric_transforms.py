import torchvision.transforms as transforms

def geometric_transforms(severity=1):
    # Severity controls the intensity of transformations
    return transforms.Compose([
        transforms.RandomRotation(degrees=30 * severity),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1 * severity, 0.1 * severity),
            scale=(1 - 0.1 * severity, 1 + 0.1 * severity),
            shear=10 * severity
        )
    ])
