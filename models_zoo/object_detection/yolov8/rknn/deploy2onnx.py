from ultralytics import YOLO


# Convert pt to onnx
weights = "/home/axuji/Downloads/yolov8.pt" # "yolov8n.pt"
model = YOLO(weights)
# Internal method written by airockchip, don't be fooled by the format name
path = model.export(format="rknn", batch=4, imgsz=640)  

