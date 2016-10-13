#!/bin/bash
NET="21class_pascal_plus_actually_pre_trained_alex_net"
ITERS=$1
GPU=''

if [ -n "$3" ]; then
   GPU="--gpu $2"
fi
mkdir -p "log/$NET"
mkdir -p "models/$NET"
LOG="log/$NET/model_${NET}_${ITERS}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
$CAFFE_ROOT/build/tools/caffe train -iterations "$ITERS" -solver "nets/$NET/solver.prototxt" -weights models/pre_trained_alex_net/bvlc_alexnet.caffemodel
