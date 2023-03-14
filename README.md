# On_the_Utility_of_Synthetic_Data
Code release for "A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation", accepted by CVPR2023.

## Requirements
- python 3.8.8
- pytorch 1.8.0
- torchvision 0.9.0

## Data preparation
1. Existing datasets. The references of the used existing datasets (e.g., VisDA-2017, ImageNet, and MetaShift) are included in the paper.
2. Our new datasets. SynSL is [here](https://pan.baidu.com/s/1Vr0xR9bu8WPHb0Ay5qU-EA?pwd=w109), our synthesized 12.8M images of 10 classes (for supervised learning, termed SynSL); SynSL-120K is [here](https://pan.baidu.com/s/10rbAZYQfST1ZhndjOpbDuQ?pwd=av1k), including our synthesized 120K images of 10 classes (train), train+SubImageNet, and three types of test data (i.e., test_IID, test_IID_wo_BG, and test_OOD); S2RDA is [here](https://pan.baidu.com/s/17C5lRDf7cpGR1kAVS2jS-Q?pwd=61tt), including two challenging transfer tasks of S2RDA-49 and S2RDA-MS-39. Please refer to the paper for more details. 

## Model training
1. Install necessary python packages.
2. Replace data paths in run.sh with those in one's own system. 
3. Run command `sh run.sh`.

The results are saved in the folder `./checkpoints/`.

## Pre-trained checkpoints
The pre-trained checkpoints for downstream synthetic-to-real classification adaptation are available here.

## Paper citation
```
@InProceedings{tang2023a,
    author    = {Tang, Hui and Jia, Kui},
    title     = {A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
}
```
