import numpy as np


# Drawing tools
def create_black(h,w,c):
    """Create Black Image
    h,w,c (int): height, width, channels of image
    """
    return np.zeros((h,w,c))

# Data tools
def analyze_class_distribution(loader, num_classes, t='train'):
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for _, masks in loader:
        for cls in range(num_classes):
            class_counts[cls] += (masks == cls).sum().item()
    print(f"Pixel count per class in {t}:", class_counts)

# Examples
if __name__ == "__main__":
    # Create black
    black = create_black(224,224,3)
    print(black.shape, np.unique(black))
    