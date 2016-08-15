#!/bin/bash
NET=$1
ITERS=$2
GPU=''

if [ -n "$3" ]; then
   GPU="--gpu $3"
fi
mkdir -p "log/$NET"
mkdir -p "models/$NET"
LOG="log/$NET/model_${NET}_${ITERS}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
python ./tools/run_model.py $NET $ITERS $GPU
