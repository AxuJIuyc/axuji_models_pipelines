import onnxruntime as ort
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from dataset import SegmentationDataset


def preprocess_image(image_path, input_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # CHW
    image = np.expand_dims(image, axis=0)  # NCHW
    return image

def postprocess_mask(output, colormap):
    # output shape: [1, num_classes, H, W]
    output = torch.from_numpy(output)
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    
    # Раскраска предсказания
    color_map = SegmentationDataset.create_color_map(colormap)
    color_map = {v:k for k,v in color_map.items()}

    seg_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        seg_img[pred == label] = color
        
    return cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

def run_onnx_inference(onnx_model_path, image_path, colormap, num_classes=6, input_size=(256, 256)):
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Preprocess image
    image_input = preprocess_image(image_path, input_size)

    # Run inference
    outputs = session.run([output_name], {input_name: image_input})
    output_mask = postprocess_mask(outputs[0], colormap)

    # Visualization
    image_vis = cv2.imread(image_path)
    image_vis = cv2.resize(image_vis, input_size)
    # color_mask = cv2.applyColorMap((output_mask * int(255 / (num_classes - 1))).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_vis, 0.5, output_mask, 0.5, 0)
    
    return image_vis, output_mask, overlay

def draw_results(image_vis, input_size, output_mask, overlay, mask_path=None, save=None):
    # Show results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 4, 1)
    plt.suptitle("ONNX")
    
    plt.title("Input Image")
    plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    if mask_path:
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, input_size)
    plt.subplot(1, 4, 2)
    plt.title("GT Mask")
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.axis("off")
        
    plt.subplot(1, 4, 4)
    plt.title("Segmentation Mask")
    plt.imshow(output_mask, cmap='nipy_spectral')
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    if save:
        plt.savefig(save)
    
    return plt

# ✅ Пример запуска:
if __name__ == "__main__":
    input_size=(224, 224)
    onnx_model_path="workdir/unet.onnx"
    image_path="../data/dataset_2/train/images/2-0_0.bmp"
    num_classes=6
    colormap = "colormap.txt"
    
    image_vis, output_mask, overlay = run_onnx_inference(
        onnx_model_path=onnx_model_path,
        image_path=image_path,
        colormap=colormap,
        num_classes=num_classes,
        input_size=input_size,
    )
    
    mask_path="../data/dataset_2/train/masks/2-0_0.png"
    plot = draw_results(image_vis, input_size, output_mask, overlay, mask_path=mask_path)
    plot.show()