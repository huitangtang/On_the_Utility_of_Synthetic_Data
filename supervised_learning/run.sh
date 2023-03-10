#!/bin/bash

python main.py  -a resnet50  --num_classes 10  --fix_dataset  --iters 2400000  --lr 0.01  --data_augmentation strong  --batch-size 128  --data_train /data/SynSL/  --data_eval /data/SynSL-120K/test_IID/ /data/SynSL-120K/test_IID_wo_BG/ /data/SynSL-120K/test_OOD/  --exp_name SynSL-PreTr  --seed 1  --evaluate_freq 20000
