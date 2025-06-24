<h1 
    align="center">Hi there, I'm Pavel 
    <img src="https://github.com/blackcater/blackcater/raw/main/images/Hi.gif" height="32"/>
</h1>

<h3 align="center">
    Computer Vision & Mashine Learning Engineer
</h3>
<h3>
    <p align="center">
        <img src=data/logo/head.jpg height="300" />
    </p>
</h3>


## Models Zoo
<!-- CLASSIFICATION -->
<details>
<summary>Classification <img src="data/logo/class.gif" height="24"/> [TO DO]</summary>

<table>
    <tr>
        <td> <img src="" height="256"/> </td>
        <td> <img src="models_zoo/classofocation/resnet50/cuda/res.jpg" height="256"/> </td>
    </tr>
</table>

<table>
    <thead>
        <tr>
            <th scope="col" rowspan="1">Model</th>
            <th scope="col" colspan="1">PyTorch</th>
            <th scope="col" colspan="1">ONNX</th>
            <th scope="col" colspan="1">OpenVino</th>
            <th scope="col" colspan="1">RK3588</th>
        </tr>
    </thead>
    <tr>
        <th><a href="models_zoo/classification/resnet50">ResNet50</a></th>
        <th>-</th><th>-</th><th>-</th><th>-</th>
    </tr>
        <tr>
        <th><a href="models_zoo/classification/EfficientNetV2">EfficientNetV2</a></th>
        <th>-</th><th>-</th><th>-</th><th>-</th>
    </tr>
</table>

</details>

<!-- OBJECT DET -->
<details>
<summary>Object Detection <img src="data/logo/od.gif" height="24"/></summary>
<table>
    <tr>
        <td> <img src="data/images/sat_1794919.jpg" height="256"/> </td>
        <td> <img src="models_zoo/object_detection/yolov8/cuda/res.jpg" height="256"/> </td>
    </tr>
</table>

<table>
    <thead>
        <tr>
            <th scope="col" rowspan="1">Model</th>
            <th scope="col" colspan="1">PyTorch</th>
            <th scope="col" colspan="1">ONNX</th>
            <th scope="col" colspan="1">OpenVino</th>
            <th scope="col" colspan="1">RK3588</th>
        </tr>
    </thead>
    <tr>
        <th><a href="models_zoo/object_detection/yolov8">YOLOv8</a></th>
        <th>+</th><th>+</th><th>+</th><th>+</th>
    </tr>
</table>
</details>

<!-- SEGMENT -->
<details>
<summary>Segmentation <img src="data/logo/seg.gif" height="24"/> [TO DO] </summary> 

<table>
    <tr>
        <td> <img src="" height="256"/> </td>
        <td> <img src="models_zoo/segmentation/pidnet/cuda/res.jpg" height="256"/> </td>
    </tr>
</table>

<table>
    <thead>
        <tr>
            <th scope="col" rowspan="1">Model</th>
            <th scope="col" colspan="1">PyTorch</th>
            <th scope="col" colspan="1">ONNX</th>
            <th scope="col" colspan="1">OpenVino</th>
            <th scope="col" colspan="1">RK3588</th>
        </tr>
    </thead>
    <tr>
        <th><a href="models_zoo/segmentation/Unet">UNet</a></th>
        <th>-</th><th>-</th><th>-</th><th>-</th>
    </tr>
    <tr>
        <th><a href="models_zoo/segmentation/pidnet">PIDNet</a></th>
        <th>-</th><th>-</th><th>-</th><th>-</th>
    </tr>
        <tr>
        <th><a href="models_zoo/segmentation/OneFormer">OneFormer</a></th>
        <th>-</th><th>-</th><th>-</th><th>-</th>
    </tr>
</table>
</details>

<!-- POSE -->
<details>
<summary>Pose Estimation <img src="data/logo/pose.gif" height="24"/> [TO DO]</summary>

</details>

<!-- ACTION -->
<details>
<summary>Action Recognition <img src="data/logo/act.gif" height="24"/> [TO DO]</summary>
</details>

## Deploing
You can [deploy](./deployment/) your models to `onnx`, `openvino` and `rknn` format