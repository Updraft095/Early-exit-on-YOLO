# Exploring YOLOv9 Optimization with Early Exit Mechanism

<!-- > [!IMPORTANT]
> This project is currently a Work In Progress and may undergo significant changes. It is not recommended for use in production environments until further notice. Please check back regularly for updates.
>
> Use of this code is at your own risk and discretion. It is advisable to consult with the project owner before deploying or integrating into any critical systems. -->

Welcome to an exploratory project on optimizing YOLOv9 model inference speed and efficiency using an early exit mechanism. This repository includes the complete codebase with the early exit mechanism, pretrained models, and detailed instructions on training and deploying YOLOv9.

## Original YOLO model (Without early exit mechanism)
- This is our addition of early exit mechanism based on the original YOLO model.
- The original YOLO model can be installed directly via pip+git:
```shell
pip install git+https://github.com/WongKinYiu/YOLO.git
yolo task.data.source=0 # source could be a single file, video, image folder, webcam ID
```

## Introduction
YOLOv9 and YOLOv7 official papers and papers related to early exit mechanism:
- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)
- [**Early Exit mechanism**: Why Should We Add Early Exits to Neural Networks?](https://doi.org/10.1007/s12559-020-09734-4)

![YOLO V9 Model Architecture based on Early Exit](https://github.com/user-attachments/assets/f4dba858-d479-4b94-aa0e-4f9955f601c9)

## Installation
To get started with our build of YOLOv9's developer mode with early exit, we recommend you clone this repository and install the required dependencies:
```shell
git clone git@github.com:Updraft095/Early-exit-on-YOLO.git
cd YOLO
pip install -r requirements.txt
```

## Dataset: Partial COCO2017
To speed up model training, we have reduced the COCO2017 dataset. If you use the reduced COCO dataset for training and validation, please make the following changes:
- Modify lines 83 and 88 in ` resize_coco.py`  to set the size of the reduced training and validation sets (you can also adjust the storage path for the reduced dataset here).
- Update the dataset path in ` yolo/config/dataset/coco.yaml`  and set the sizes of the reduced training and validation sets to match the configurations in ` resize_coco.py` .

## Model Structure Modification: Adding Early Exit Layers
To integrate an early exit mechanism into the model, follow these configuration steps:
1. Copy an existing model structure YAML file in the `yolo/config/model` directory and modify it.
2. Configure three ` EarlyExitSampler`  layers and one ` EarlyExitMultiheadDetection`  layer for each early exit. Ensure that the parameters for each early exit layer, especially ` in_channels` , are adjusted to match the output of the preceding layer. Example configuration:
```shell
- EarlyExitSampler:
    args: {in_channels: 32, out_channels: 256, output_size: 80}
- EarlyExitSampler:
    args: {in_channels: 32, out_channels: 512, output_size: 40}
- EarlyExitSampler:
    args: {in_channels: 32, out_channels: 1024, output_size: 20}
- EarlyExitMultiheadDetection:
    args:
      version: v7
    source: [-3, -2, -1]
```
3. Adjust the source of the subsequent layers, typically skipping over the early exit layers to ensure correct layer hierarchy.



## Features

<table>
<tr><td>

| Tools | pip 🐍 | HuggingFace 🤗 | Docker 🐳 |
| -------------------- | :----: | :--------------: | :-------: |
| Compatibility       | ✅     | ✅               | 🧪        |

|  Phase    | Training | Validation | Inference |
| ------------------- | :------: | :---------: | :-------: |
| Supported           | ✅       | ✅          | ✅        |

</td><td>

| Device | CUDA       | CPU       | MPS       |
| ------------------ | :---------: | :-------: | :-------: |
| PyTorch            | v1.12      | v2.3+     | v1.12     |
| ONNX               | ✅         | ✅        | -         |
| TensorRT           | ✅         | -        | -         |
| OpenVINO           | -          | 🧪        | ❔        |

</td></tr> </table>



## Task
These are simple examples. In this project, we aim to optimize the YOLOv9 model’s inference speed and efficiency by integrating five early exit points into the model. These early exits allow the model to make predictions at various stages, reducing the need to process the entire network for simpler tasks and speeding up the inference process. Through this setup, we will evaluate and compare the model’s performance and efficiency across different exit points, providing insights into the trade-offs between accuracy and inference time.

## Training
To train YOLO with early exit mechanism on your machine/dataset:

1. If you use other datasets, please modify the configuration file `yolo/config/dataset/**.yaml` to point to your dataset. If you use our pruned coco dataset, please run `resize_coco.py` first.
2. Set batch_size and epoch by using `task.data.batch_size` and `task.epoch`.
3. Use name to specify a custom name for this training experiment (optional).
4. Training will be done in two phases: For the first 50% of epochs, YOLO will be fully trained. After that, the weights will be frozen and only the early dropout layers will be trained.
5. Run the training script:
```shell
python yolo/lazy.py task=train model=v9_c dataset=coco weight=False 
use_wandb=False task.data.batch_size=8 task.epoch=50 name=early
```

[//]: # (### Transfer Learning)

[//]: # (To perform transfer learning with YOLOv9:)

[//]: # (```shell)

[//]: # (python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c dataset={dataset_config} device={cpu, mps, cuda})

[//]: # (```)

### Inference
To use the YOLO model with early exit for object detection on your machine/dataset:

1. Specify the inference data path using `task.data.source` (e.g., PNG images).
2. Set the path to the trained weights with `weight`.
3. Use `model.early_exit.specified_layer` to select which early exit layer to use (fixed early exit layer, regardless of exit mechanism).
4. Configure the exit mechanism type via `model.early_exit.dynamic`, choosing between `entropy` (Entropy-based early exit) or `confidence` (Confidence-based early exit).
5. Set the threshold for the early exit mechanism with `model.early_exit.confidence`. If the value at a given layer exceeds this threshold, inference will exit at that layer.
6. Run the reference script:
```shell
python yolo/lazy.py task=inference weight=runs/train/early/weights/E049.pt 
task.data.source=/home/YOLO/demo/images/inference model.early_exit.specified_layer=4
```

### Validation
To validate model performance:
```shell
python yolo/lazy.py task=validation weight=runs/train/early/weights/E049.pt 
task.data.source=coco model.early_exit.specified_layer=4
```
