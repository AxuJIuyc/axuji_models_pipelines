from ultralytics import YOLO


# Convert pt to onnx
weights = "crossroads_yolov8n.pt" # "yolov8n.pt"
model = YOLO(weights)
# Internal method written by airockchip, don't be fooled by the format name
path = model.export(format="rknn")  

