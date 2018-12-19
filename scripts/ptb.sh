#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${1}
cd ..
python3 NJUParser/trainer/distance_trainer.py  --filename ${2} --configpath ${3}