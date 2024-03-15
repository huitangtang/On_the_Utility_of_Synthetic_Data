# On_the_Utility_of_Synthetic_Data
Code release for "A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation", accepted by CVPR2023.

[Project Page](https://huitangtang.github.io/On_the_Utility_of_Synthetic_Data/) $\cdot$ [PDF Download](http://arxiv.org/abs/2303.09165) $\cdot$ [Dataset Download](https://cove.thecvf.com/datasets/892)


One can get access to our datasets via Google Drive (above) or Baidu Netdisk (below).

## Requirements
- python 3.8.8
- pytorch 1.8.0
- torchvision 0.9.0

## Data preparation
1. Existing datasets. The references of the used existing datasets (e.g., ShapeNet, VisDA-2017, ImageNet, MetaShift, and ObjectNet) are included in the paper.
2. Our new datasets. One can download them from the link above; otherwise, download them from the link below. SynSL is [here](https://pan.baidu.com/s/1Vr0xR9bu8WPHb0Ay5qU-EA?pwd=w109), our synthesized 12.8M images of 10 classes (for supervised learning, termed SynSL); SynSL-120K is [here](https://pan.baidu.com/s/10rbAZYQfST1ZhndjOpbDuQ?pwd=av1k), including our synthesized 120K images of 10 classes (train), train+SubImageNet, and three types of test data (i.e., test_IID, test_IID_wo_BG, and test_OOD); **S2RDA is [here](https://pan.baidu.com/s/1fHHaqrEHbUZLXEg9XKpgSg?pwd=w9wa), including two challenging transfer tasks of S2RDA-49 and S2RDA-MS-39.** Please refer to the paper for more details.
3. The validation and test splits of the real domains in S2RDA are also provided in this repository.

## Model training
1. Install necessary python packages.
2. Replace data paths in run.sh with those in one's own system. 
3. Run command `sh run.sh`.

The results are saved in the folder `./checkpoints/`.

## Pre-trained checkpoints
The pre-trained checkpoints for downstream synthetic-to-real classification adaptation are available [here](https://drive.google.com/drive/folders/1g4uqLxlgC2txarIgJqSbpI8OAkgkjG4V?usp=sharing) or [there](https://pan.baidu.com/s/1Oj1EubGWHOn_Hz8B8Dfh0g?pwd=9x2y), and they are obtained at the last pre-training iteration.

## Paper citation
```
@InProceedings{tang2023new,
    author    = {Tang, Hui and Jia, Kui},
    title     = {A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
}
```
