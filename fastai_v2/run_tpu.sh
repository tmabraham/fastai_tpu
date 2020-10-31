#!/bin/bash
# Setup environment and run fastai training code on TPU
TPU_IP_ADDRESS="10.9.54.194"
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
python tpu_distributed_fastai2.py
