
import cv2
import torch

import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from unet import UNet
from dataset import SegmentationDataset


# -------- Инференс --------
def preprocess(image_path, device, model_size=(256,256)):
    image = cv2.imread(image_path)
    orig = image.copy()
    orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, model_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = T.ToTensor()(image).unsqueeze(0).to(device)
    return orig, image

def inference(model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return pred

def draw_results(pred, image, colormap, mask_path=None, save=None):
    # image = image.squeeze(0).cpu().numpy()
    # Show results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.suptitle("PyTorch")
    
    
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis("off")
    
    if mask_path:
        mask = cv2.imread(mask_path)
        # mask = cv2.resize(mask, input_size)
        plt.subplot(1, 3, 2)
        plt.title("GT Mask")
        plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
    # Раскраска предсказания
    color_map = SegmentationDataset.create_color_map(colormap)
    color_map = {v:k for k,v in color_map.items()}

    seg_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        seg_img[pred == label] = color

    seg_img = cv2.resize(seg_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(seg_img, cmap='nipy_spectral')
    plt.axis("off")
    
    if save:
        print("Saved: segmentation_result.png")
        plt.savefig(save)    
    return plt

    
if __name__ == "__main__":
    import os

    weights = "/home/axuji/Projects/Segmentation/Unet/workdir/run_4/unet_trained.pth"
    num_classes=2
    colormap="/home/axuji/Projects/SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/colormap.txt"
    model_size=(32,24)
    device = 'cuda'
    figs_save = "workdir"
    
    img_dir = "/home/axuji/Projects/SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/val/images"
    mask_dir = "/home/axuji/Projects/SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/val/masks"
    # image_path = "/home/axuji/Projects/SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/val/images/003d_3.png"
    # mask_path = "/home/axuji/Projects/SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/val/masks/003d_3.png"
    
    model = UNet(out_channels=num_classes)
    model.load_state_dict(torch.load(weights, weights_only=True))
    
    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))
    place = "test_"
    i = 0
    places = os.listdir(figs_save)
    while place+str(i) in places:
        i += 1
    place = os.path.join(figs_save, place+str(i))
    os.makedirs(place, exist_ok=True)
    for img, mask in zip(images, masks):
        image_path = os.path.join(img_dir, img)
        mask_path = os.path.join(mask_dir, mask)
        orig, im = preprocess(image_path, device, model_size)
        results = inference(model, im)
        plot = draw_results(results, orig, colormap, mask_path)
        # plot.show()
        plot.savefig(os.path.join(place, img))