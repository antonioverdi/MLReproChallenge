<div align="center">    
 
# ML Reproducibility Challenge 2020   
</div>

## Description

This repository contains all of the code used to replicate the results of the ICLR 2020 conference paper

[___"Comparing Rewinding and Fine-Tuning in Neural Network Pruning"___](https://arxiv.org/pdf/2003.02389.pdf)

It also includes code for unstructured sensitivity based pruning from:
[___"SNIP: Single-Shot Network Pruning Based on Connection Sensitivity"___](https://arxiv.org/pdf/1810.02340.pdf)

## Requirements/Setup

## Instructions for Running Code

## Architectures and Datasets

| Dataset | Network | Optimizer | Learning Rate |
| :----------------: | :----------------: | :----------------: | :----------------: |
| CIFAR 10 | ResNet-56 | Nesterov SGD <br> <img src="https://render.githubusercontent.com/render/math?math=\beta = 0.9"> <br> Batch Size: 128 <br> Weight Decay: 0.0002 <br> Epochs: 182| <img src="https://render.githubusercontent.com/render/math?math=\alpha = 0.1 \quad \forall t \in [0,91)"> <br> <img src="https://render.githubusercontent.com/render/math?math=\alpha = 0.01 \quad \forall t \in [91, 136)"> <br> <img src="https://render.githubusercontent.com/render/math?math=\alpha = 0.001 \quad \forall t \in [136, 182]">|
| ImageNet | ResNet-34 | Nesterov SGD <br> <img src="https://render.githubusercontent.com/render/math?math=\beta = 0.9"> <br> Batch Size: 1024 <br> Weight Decay: 0.0001 <br> Epochs: 90 |  |
| ImageNet | Resnet-50 | Nesterov SGD <br> <img src="https://render.githubusercontent.com/render/math?math=\beta = 0.9"> <br> Batch Size: 1024 <br> Weight Decay: 0.0001 <br> Epochs: 90  |  |
| WMT16 EN-DE | GNMT | Adam <br> <img src="https://render.githubusercontent.com/render/math?math=\beta_1 = 0.9"> <br> <img src="https://render.githubusercontent.com/render/math?math=\beta_2 = 0.999"> <br> Batch Size: 2048 <br> Epochs: 5  |  |
