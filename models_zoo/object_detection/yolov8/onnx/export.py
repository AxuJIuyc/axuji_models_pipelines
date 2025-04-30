from ultralytics import YOLO



modelpath = "../rknn/crossroads_yolov8n.pt"
# savepath = "yolo.onnx"
model = YOLO(modelpath)
model.export(format='onnx', int8=False,)

