import torch
import torch.nn as nn
import torch.profiler
import torchvision.models as models



def profile(model=None, inputs=(224,224,3), steps=2):
    if not model:
        model = models.resnet18().cuda()
    inputs = torch.randn(1, inputs[2], inputs[0], inputs[1]).cuda()
    inputs /= 255.0
    model.eval()

    wait=1 
    warmup=1 
    active=steps
    repeat=1
    steps = (wait + warmup + active) * repeat
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],  # Профилируем и CPU и GPU
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir'),
        record_shapes=True,
        # with_stack=True
    ) as prof:
        for _ in range(steps):
            with torch.profiler.record_function("model_inference"):
                model(inputs)
            prof.step()  # обязательно вызывать!

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("../models_zoo/object_detection/yolov8/cuda/yolov8_crossroads_260525.pt").cuda()
    
    profile(model=model, inputs=(256, 256, 3), steps=2)