# UTransformer: Semantic Segmentation with PyTorch
![alt text](https://github.com/MoaazK/comp511-project/blob/master/assets/Result.png?raw=true)

Implementation of the [U-Net Transformer](https://arxiv.org/abs/2103.06104) and comparison with [Attention U-Net](https://arxiv.org/abs/1804.03999) and baseline [U-Net](https://arxiv.org/abs/1505.04597)

- [Quick Start](#quick-start)
- [Description](#description)
- [Requirements](#requirements)

## Quick Start
```bash
conda create -n <environment_name> python=3.8
conda activate <enviroment_name>
pip install -r requirements.txt
```
## Description
This model was trained on [Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) Kaggle dataset and scored a Dice Score of 0.8925 on validation data and 0.8868 on test data.

This model can easily be extended for multiclass classification. Moreover, I wrote U-Net generlized implementation which can be easily be extended for different types of U-Nets.

## Requirements
1. [Anaconda](https://www.anaconda.com/products/distribution)
2. [CUDA 11.3 or later](https://developer.nvidia.com/cuda-downloads)
3. [PyTorch 1.12 or later](https://pytorch.org/get-started/locally/)
4. [Jupyter Notebook](https://jupyter.org/)
5. 8 GPUs to run [U-Net Transformer](https://arxiv.org/abs/2103.06104)
6. 
