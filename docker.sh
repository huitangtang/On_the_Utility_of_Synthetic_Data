#!/bin/bash 
#nvidia-docker run  --privileged  --shm-size=51200m  --network host  -it  pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime   /bin/bash

apt-get update && apt --fix-broken install -y && apt-get install sshfs -y 
apt-get update && apt --fix-broken install -y && apt install -y libgl1-mesa-dev libglib2.0-0 && pip install opencv-python
pip install scipy && pip install ipdb && pip install tensorboard && pip install einops #&& pip install matplotlab
