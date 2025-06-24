import torch
from unet import UNet  # Импорт твоей модели

import run_onnx
import run_torch


def export(
    model_path, 
    model_size=(256, 256), 
    out_channels=6, 
    save_path="unet.onnx",
    opset_version=11
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=out_channels).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = torch.randn(1, 3, *model_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        do_constant_folding=True,
        # dynamic_axes={
        #     "input": {0: "batch_size"},
        #     "output": {0: "batch_size"}
        # }
    )

    print("✅ UNet успешно экспортирован в ONNX: unet.onnx")
    return save_path
    
def check(
    onnx_model_path, 
    pt_model_path, 
    image_path, 
    num_classes, 
    colormap, 
    mask_path=None, 
    input_size=(256,256), 
    device='cpu'
    ):
    # Проверка корректности модели
    # ONNX
    image_vis, output_mask, overlay = run_onnx.run_onnx_inference(
        onnx_model_path=onnx_model_path,
        image_path=image_path,
        colormap=colormap,
        num_classes=num_classes,
        input_size=input_size
    )
    plot = run_onnx.draw_results(image_vis, input_size, output_mask, overlay, mask_path=mask_path)
    plot.show()
    
    # PyTorch    
    model = UNet(out_channels=num_classes)
    model.load_state_dict(torch.load(pt_model_path, weights_only=True))
    orig, im = run_torch.preprocess(image_path, device, input_size)
    results = run_torch.inference(model, im)
    plot = run_torch.draw_results(results, orig, colormap, mask_path)
    plot.show()
    

if __name__ == "__main__":
    model_path = "workdir/run_4/unet_trained.pth"
    model_size = (32,32)
    num_classes = 2
    
    onnx_model_path = export(
        model_path, 
        model_size=model_size, 
        out_channels=num_classes,
        save_path="workdir/run_4/unet.onnx"
    )
    
    # Compare weights
    pt_model_path = model_path
    image_path = "/home/axuji/Projects/SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/train/images/002d_3.png"
    mask_path = "/home/axuji/Projects/SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/train/masks/002d_3.png"
    colormap="/home/axuji/Projects/SpaceOrient/dataset/task_part_1_dataset_2025_06_11_05_12_35_segmentation mask 1.1/colormap.txt"
    device = 'cuda'

    check(
        onnx_model_path, 
        pt_model_path, 
        image_path, 
        num_classes, 
        colormap, 
        mask_path=mask_path, 
        input_size=model_size, 
        device=device      
    )