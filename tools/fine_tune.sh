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
$CAFFE_ROOT/build/tools/caffe train -solver nets/person_vs_background_vs_random_pre_trained_alex_net/solver.prototxt -weights models/pre_trained_alex_net/bvlc_alexnet.caffemodel -gpu 0
