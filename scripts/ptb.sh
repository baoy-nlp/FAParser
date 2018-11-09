#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1}
cd ../..
python3 examples/main.py --config_files configs/ptb.yaml --mode ${2} --exp_name ${3}