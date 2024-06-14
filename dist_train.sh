#!/usr/bin/env bash
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
CONFIG='configs/lup_dg.yaml'
GPUS=4
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    train.py --cfg $CONFIG ${@:3} #--resume --ckpt /dockerdata/path/log/SeqNet-DG/LUP/epoch_0.pth 
