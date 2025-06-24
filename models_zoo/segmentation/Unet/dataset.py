import os

from matplotlib import axes
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import random

import tools


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=224, colormap='colormap.txt'):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        checksum(self.image_paths, self.mask_paths)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])
        self.image_size = image_size
        # Заранее создаём таблицу соответствия цветов и классов
        self.color_to_class = self.create_color_map(colormap)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.imread(self.mask_paths[idx])
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        mask = self.rgb_to_class(mask_rgb)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).long()
        return img, mask
    
    @staticmethod
    def create_color_map(colormap):
        with open(colormap, 'r') as f:
            text = f.read()
        rows = text.splitlines()
        i = 0
        colors = {(0,0,0):0}
        for row in rows:
            color = tuple(map(int, row.split(',')))
            if color not in colors:
                colors.update({color:i})
            i += 1
        return colors  
    
    def rgb_to_class(self, mask_rgb):
        h, w, _ = mask_rgb.shape
        mask_class = np.zeros((h, w), dtype=np.uint8)
        for color, idx in self.color_to_class.items():
            mask_class[(mask_rgb == color).all(axis=2)] = idx
        return mask_class
    
def checksum(imgs, masks):
    assert len(imgs) == len(masks), "Количество изображений и масок не совпадает"
    
    for i,m in zip(imgs, masks):
        i_name, _ = os.path.splitext(os.path.basename(i))
        m_name, _ = os.path.splitext(os.path.basename(m))
        assert i_name == m_name, f"Порядок изображений не совпадает: {i} & {m}"
    return True

def prepare_dataset(img_dir, mask_dir, save_folder, ratio=(9,1), add_void=False, maskext='.png'):
    """Split data to dataset

    Args:
        img_dir (_type_): path to images
        mask_dir (_type_): path to mask-images
        save_folder (_type_): path to save folder
        ratio (tuple): Train:Val split ratio. Defaults to (9,1).
        add_void (bool) Defaults to False
    """
    os.makedirs(save_folder, exist_ok=True)
    train_im = os.path.join(save_folder, "train", "images")
    val_im = os.path.join(save_folder, "val", "images")
    train_mask = os.path.join(save_folder, "train", "masks")
    val_mask = os.path.join(save_folder, "val", "masks")
    
    for f in [train_im, val_im, train_mask, val_mask]:
        os.makedirs(f, exist_ok=True)
        
    images = os.listdir(img_dir)
    masks = os.listdir(mask_dir)
    random.seed(0)
    random.shuffle(images)
    
    train_part = len(images)//(ratio[0]+ratio[1])*ratio[0]
    train = images[:train_part]
    val = images[train_part:]
    
    for i, part in zip(['train', 'val'], [train, val]):
        for im in part:
            s_im_p = os.path.join(img_dir, im) # source_image_path
            
            name, ext = os.path.splitext(im)
            mask = name+maskext
            d_m_p = os.path.join(save_folder, i, 'masks', mask) # dest_mask_path
            if mask not in masks:
                if add_void:
                    h,w,c = cv2.imread(s_im_p).shape
                    black = tools.create_black(h,w,c)
                    cv2.imwrite(d_m_p, black)
                else:
                    continue
            else:
                s_m_p = os.path.join(mask_dir, mask) # source_mask_path
                os.system(f"cp '{s_m_p}' '{d_m_p}'")
            d_im_p = os.path.join(save_folder, i, 'images', im) # source_image_path
            os.system(f"cp '{s_im_p}' '{d_im_p}'")
            
                
    print("Train Images:", len(os.listdir(train_im)))
    print("Train Masks:", len(os.listdir(train_mask)))
    print("Val Images:", len(os.listdir(val_im)))
    print("Val Masks:", len(os.listdir(val_mask)))
                
if __name__ == "__main__":
    img_dir = "../../SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/task/JPEGImages/" 
    mask_dir = "../../SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/task/SegmentationClass/"
    save_folder = "../../SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/"
    ratio=(8,1) 
    add_void=True 
    maskext='.png'
    
    prepare_dataset(img_dir, mask_dir, save_folder, ratio, add_void, maskext)