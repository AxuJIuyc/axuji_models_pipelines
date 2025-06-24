import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from unet import UNet
from dataset import SegmentationDataset
import tools

# -------- Обучение --------
def update_confusion_matrix(preds, targets, num_classes, cm):
    """
    preds: [B, H, W] - предсказанные классы
    targets: [B, H, W] - истинные классы
    """
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    cm += confusion_matrix(targets, preds, labels=range(num_classes))
    return cm

def compute_metrics(cm):
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    epsilon = 1e-7
    IoU = TP / (TP + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    accuracy = TP.sum() / cm.sum()

    metrics = {
        'accuracy': accuracy.item(),
        'IoU': IoU,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return metrics

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, masks in tqdm(loader, desc="🔁 Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            cm = update_confusion_matrix(preds, masks, num_classes, cm)

    metrics = compute_metrics(cm)
    return total_loss / len(loader), metrics

def validate_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="🔎 Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            cm = update_confusion_matrix(preds, masks, num_classes, cm)

    metrics = compute_metrics(cm)
    return total_loss / len(loader), metrics

def plot_training_history(history, save, best):
    
    for x in history.keys():
        if x in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]:
            continue
        history[x] = np.array(history[x]).T
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Average
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.axvline(x=best, color ='red')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_accuracy'], label='Train')
    plt.plot(epochs, history['val_accuracy'], label='Val')
    plt.axvline(x=best, color ='red')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['train_IoU'].mean(axis=0), label='Train')
    plt.plot(epochs, history['val_IoU'].mean(axis=0), label='Val')
    plt.axvline(x=best, color ='red')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['train_precision'].mean(axis=0), label='Train')
    plt.plot(epochs, history['val_precision'].mean(axis=0), label='Val')
    plt.axvline(x=best, color ='red')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['train_recall'].mean(axis=0), label='Train')
    plt.plot(epochs, history['val_recall'].mean(axis=0), label='Val')
    plt.axvline(x=best, color ='red')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs, history['train_f1'].mean(axis=0), label='F1')
    plt.plot(epochs, history['val_f1'].mean(axis=0), label='F1')
    plt.axvline(x=best, color ='red')
    plt.title('F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save,"results.png"))
    
    # Per class
    plt.figure(figsize=(20, 9))
    
    plt.subplot(2, 4, 1)
    for i, x in enumerate(history['train_IoU']):
        plt.plot(epochs, x, label=i)
    plt.axvline(x=best, color ='red')
    plt.title('train_IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.subplot(2, 4, 2)
    for i, x in enumerate(history['val_IoU']):
        plt.plot(epochs, x, label=i)
    plt.axvline(x=best, color ='red')
    plt.title('val_IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.subplot(2, 4, 3)
    for i, x in enumerate(history['train_f1']):
        plt.plot(epochs, x, label=i)
    plt.axvline(x=best, color ='red')
    plt.title('Train F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    
    plt.subplot(2, 4, 4)
    for i, x in enumerate(history['val_f1']):
        plt.plot(epochs, x, label=i)
    plt.axvline(x=best, color ='red')
    plt.title('Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    
    plt.subplot(2, 4, 5)
    for i, x in enumerate(history['train_precision']):
        plt.plot(epochs, x, label=i)
    plt.axvline(x=best, color ='red')
    plt.title('Train Precision')
    plt.xlabel('Epoch')
    plt.ylabel('P')
    plt.legend()
    
    plt.subplot(2, 4, 6)
    for i, x in enumerate(history['val_precision']):
        plt.plot(epochs, x, label=i)
    plt.axvline(x=best, color ='red')
    plt.title('Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('P')
    plt.legend()
    
    plt.subplot(2, 4, 7)
    for i, x in enumerate(history['train_recall']):
        plt.plot(epochs, x, label=i)
    plt.axvline(x=best, color ='red')
    plt.title('Train Recall')
    plt.xlabel('Epoch')
    plt.ylabel('R')
    plt.legend()
    
    plt.subplot(2, 4, 8)
    for i, x in enumerate(history['val_recall']):
        plt.plot(epochs, x, label=i)
    plt.axvline(x=best, color ='red')
    plt.title('Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('R')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save,"results_percls.png"))
    
def train_model(
    model, 
    train_loader, 
    val_loader, 
    num_classes, 
    device, 
    epochs=10, 
    lr=1e-3,
    aim_score='f1'
    ):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    history = {}
    for x in ['train', 'val']:
        history.update({
            x+"_loss":[],
            x+"_accuracy":[],
            x+"_IoU":[],
            x+"_precision":[],
            x+"_recall":[],
            x+"_f1":[]
        })

    best = (model.state_dict(), 0)
    last_aim_score = 0
    for epoch in range(1, epochs + 1):
        print(f"\n🌟 Epoch {epoch}/{epochs}")
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, num_classes)

        print(f"📉 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        for i in range(num_classes):
            print(f"   Класс {i}: IoU={val_metrics['IoU'][i]:.3f} | F1={val_metrics['f1'][i]:.3f}")

        mean_aim_score = val_metrics[aim_score].mean()
        if mean_aim_score > last_aim_score:
            last_aim_score = mean_aim_score
            best = (model.state_dict(), epoch)
            
        history = collect_history_metrics(history, (train_loss, val_loss, train_metrics, val_metrics))

    print("✅ Обучение завершено. Отрисовываю графики...")
    print("💾 Модель сохранена в 'unet_trained.pth'") 
    workdir = "workdir"
    os.makedirs(workdir, exist_ok=True)
    place = "run_"
    i = 0
    places = os.listdir(workdir)
    while place+str(i) in places:
        i += 1
    save_dir = os.path.join(workdir, place+str(i))
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(best[0], os.path.join(save_dir, "unet_trained.pth"))
    print("Лучшая эпоха:", best[1])
    plot_training_history(history, save_dir, best[1])

def collect_history_metrics(history, metrics):
    train_loss, val_loss, train_metrics, val_metrics = metrics

    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_metrics['accuracy'])
    history['train_IoU'].append(train_metrics['IoU'])
    history['train_precision'].append(train_metrics['precision'])
    history['train_recall'].append(train_metrics['recall'])
    history['train_f1'].append(train_metrics['f1'])    
    
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_metrics['accuracy'])
    history['val_IoU'].append(val_metrics['IoU'])
    history['val_precision'].append(val_metrics['precision'])
    history['val_recall'].append(val_metrics['recall'])
    history['val_f1'].append(val_metrics['f1'])
    return history

if __name__ == "__main__":
    # N = 3
    # train_im = f"/home/axuji/Projects/Segmentation/data/dataset_{N}/train/images"
    # val_im = f"/home/axuji/Projects/Segmentation/data/dataset_{N}/val/images"
    # train_masks = f"/home/axuji/Projects/Segmentation/data/dataset_{N}/train/masks"
    # val_masks = f"/home/axuji/Projects/Segmentation/data/dataset_{N}/val/masks"
    
    folder = "/home/axuji/Projects/SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1"
    train_im = os.path.join(folder, "train/images")
    val_im = os.path.join(folder, "val/images")
    train_masks = os.path.join(folder, "train/masks")
    val_masks = os.path.join(folder, "val/masks")
    batch = 16
    
    # colormap = "/home/axuji/Projects/Segmentation/Unet/colormap.txt"
    colormap = os.path.join(folder, "colormap.txt")
    
    num_classes = 2 # with background
    img_size = 32
    
    model = UNet(out_channels=num_classes)
    train_dataset = SegmentationDataset(train_im, train_masks, image_size=img_size, colormap=colormap)
    val_dataset = SegmentationDataset(val_im, val_masks, image_size=img_size, colormap=colormap)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    tools.analyze_class_distribution(train_loader, num_classes, t='Train')
    tools.analyze_class_distribution(val_loader, num_classes, t='Val')
    print("Классы:", (train_dataset.color_to_class))
    train_model(model, train_loader, val_loader, num_classes=num_classes, device='cuda', epochs=40)
