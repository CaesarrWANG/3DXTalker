#!/bin/bash


echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
export CUDA_VISIBLE_DEVICES=0,1
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-Not Set}"

accelerate launch --config_file ./accelerate_configs/accelerate_gpus.yaml --mixed_precision no --multi_gpu train.py > train_TerminalLog.log 2>&1 & 
