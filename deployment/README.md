<h1 
    align="center">This directory is dedicated to install deploying tools 
    <img src="https://github.com/blackcater/blackcater/raw/main/images/Hi.gif" height="32"/>
</h1>

# PyTorch
# ONNX
# OpenVINO
# RKNN

## 1. Desktop processes
### Environment
```
conda create -n rknn python==3.10
conda activate rknn
```
### Install requirements
```
# Download
wget https://github.com/airockchip/rknn-toolkit2/raw/v2.1.0/rknn-toolkit2/packages/requirements_cp310-2.1.0.txt

# Install
pip install -r requirements_cp310-2.1.0.txt
```
### Install whls for rknn-toolkit2
```
# Download
wget https://github.com/airockchip/rknn-toolkit2/raw/v2.1.0/rknn-toolkit2/packages/rknn_toolkit2-2.1.0+708089d1-cp310-cp310-linux_x86_64.whl

# Install
pip install rknn_toolkit2-2.1.0+708089d1-cp310-cp310-linux_x86_64.whl
```

## 2. RKNN-machine processes
### Install OS
1. Download image:

      | [Ubuntu (OrangePi 5)](https://drive.google.com/drive/folders/1i5zQOg1GIA4_VNGikFl2nPM0Y2MBw2M0) | [Ubuntu (OrangePi 5B)](https://drive.google.com/drive/folders/1xhP1KeW_hL5Ka4nDuwBa8N40U8BN0AC9) | [Armbian (OrangePi 5/5B)](https://www.armbian.com/orangepi-5/) |
      | :---: | :---: | :---: |

  2. Burn it to SD card.

  3. Plug SD card to Orange Pi.

### Configure OrangePi for running models

  1. Update [**librknnrt.so**](https://github.com/airockchip/rknn-toolkit2/blob/v2.1.0/rknpu2/runtime/Linux/librknn_api/aarch64/).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/v2.1.0/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

      # Move to /usr/lib
      sudo mv ./librknnrt.so /usr/lib
      ```

  2. Install whls for [**rknn-toolkit-lite2**](https://github.com/airockchip/rknn-toolkit2/tree/v2.1.0/rknn-toolkit-lite2/packages).

      ```
      # Download
      wget https://github.com/airockchip/rknn-toolkit2/raw/v2.1.0/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl

      # Install
      pip install rknn_toolkit_lite2-2.1.0-cp310-cp310-linux_aarch64.whl
      ```

  3. Install opencv-python and other requirements (if necessary).

      ```
      pip install opencv-python
      ```