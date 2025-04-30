from ultralytics import YOLO


# Convert pt to onnx
weights = "crossroads_roofs-yolov8m-1.12.2.3.4.9_0(best).pt" # "yolov8n.pt"
model = YOLO(weights)
# Internal method written by airockchip, don't be fooled by the format name
path = model.export(format="rknn")  

