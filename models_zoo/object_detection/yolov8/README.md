<h1 align="center">
    This directory is dedicated to running, training and deploying the 
    <a 
        href="https://github.com/ultralytics/ultralytics"> YOLO 
    </a> 
    model
    <img 
        src="https://github.com/blackcater/blackcater/raw/main/images/Hi.gif" height="32"
    />
</h1>

<table>
    <tr>
        <td> <img src="../../../data/images/sat_1794919.jpg" height="256"/> </td>
        <td> <img src="cuda/res.jpg" height="256"/> </td>
    </tr>
</table>

# 1. Install
## 1.1 Environment
```bash
conda create -n yolov8det python==3.12
conda activate yolov8det
```
## 1.2 Get Repo
```bash
pip install ultralytics
```

# 2. Get Started
Edit `cuda/inference.py` file: check path to your `weights` and your `image`.  
Run code:
```bash
python cuda/inference.py
```

# 3. Train
You need to edit the file `train/train.py` and run it.
```bash
python train/train.py
```

# 4. Deploy
To convert a model to one of the following formats, first install the required [dependencies](../../../deployment/README.md). Then follow these steps:

## 4.1 ONNX

## 4.2 RKNN
Go to "rknn" environment if it has been created.

### Special install
Install ultralytics_yolov8 special for converting pt -> onnx (optimized for rknn).
```
# Clone repo
git clone https://github.com/airockchip/ultralytics_yolov8

# Go to cloned directory
cd ultralytics_yolov8

# Install as package
pip install -e .

# Edit cfg file
nano ultralytics/cfg/default.yaml
"""
  model: {path/to/model.pt}
  batch: 1
  imgsz: {your img size}
"""
```

### Convert model to .rknn
#### 4.2.1. Convert pt to onnx
```
cd rknn
python deploy2onnx.py
```
Check the input size when exporting the model. If necessary, change batch_size parameter in ultralytics/cfg/default.yaml to any value.

You should get ***9 outputs***. Check model.onnx in netron.app.

#### 4.2.2. Convert onnx to rknn
```
# Clone repo
git clone https://github.com/airockchip/rknn_model_zoo

# Go to directory with converter
cd rknn_model_zoo/examples/yolov8/python

# Edit `convert.py`
DATASET_PATH = "path/to/dataset.txt"
DEFAULT_RKNN_PATH = "path/to/model.rknn"

# Run converter
python convert.py <path-to-onnx-model>/yolov8n.onnx rk3588 i8 ../model/yolov8n.rknn
```
If the model has issues or warnings in convertation process, you can change opset version from 12 to 17 or 19, depending on PyTorch version. Currently, RKNN==2.1.0 recommends opset 19.
#### 4.2.3. Save and send it to Orange Pi

#### 4.2.4. Fast Start on RKNN
```
git clone https://github.com/Applied-Deep-Learning-Lab/rk3588-yolov8.git
cd rk3588-yolov8
python main.py 

# This correct for video
# If you need check image, you need edit main.py and base.py 
```

#### Check NPU utilization
```
watch sudo cat /sys/kernel/debug/rknpu/load
```

## 4.3 OpenVINO

# 5. Metrics
`CPU desktop - Intel Core i9-14900HX`  
`GPU desktop - RTX 4070 mobile`  
`RKNN platform - RK3588 (Orange-Pi 16GB)`  

<style>
td, th {
   border: 2px solid;
}
</style>

## 5.1 Performance, FPS
Without batching  
Size: `256x256` pxls
<table>
  <tr>
    <th>Model</th>
    <th>CPU (torch)</th>
    <th>GPU (torch)</th>
    <th>CPU (openvino)</th>
    <th>CPU (onnx)</th>
    <th>GPU (onnx)</th>
    <th>CPU (orange, torch)</th>
    <th>NPU (orange, int) </th>
    <th>NPU (orange, fp) </th>
  </tr>
  <tr>
    <td>yolov8n</td>
    <td>87</td>
    <td>140</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>2.5</td>
    <td>65</td>
    <td>58</td>
  </tr>
  <tr>
    <td>yolov8m</td>
    <td>33</td>
    <td>112</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>0.76</td>
    <td>38</td>
    <td>21</td>
  </tr>
</table>

## 5.2 Correctness
`(A)ccuracy, (P)recision, (R)ecall` for validation data

<table>
    <thead>
        <tr>
            <th scope="col" rowspan="2">Model</th>
            <th scope="col" colspan="3">torch</th>
            <th scope="col" colspan="3">onnx</th>
            <th scope="col" colspan="3">rknn</th>
            <th scope="col" colspan="3">openvino</th>
        </tr>
        <tr>
            <th scope="col">A</th>
            <th scope="col">P</th>
            <th scope="col">R</th>
            <th scope="col">A</th>
            <th scope="col">P</th>
            <th scope="col">R</th>
            <th scope="col">A</th>
            <th scope="col">P</th>
            <th scope="col">R</th>
            <th scope="col">A</th>
            <th scope="col">P</th>
            <th scope="col">R</th>
        </tr>
    </thead>
  <tr>
    <td>yolov8n</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
  <tr>
    <td>yolov8m</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
    <td>N/A</td>
  </tr>
</table>