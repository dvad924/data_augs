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
$CAFFE_ROOT/build/tools/caffe train -solver nets/person_vs_background_vs_random_pre_trained_alex_net/solver.prototxt -snapshot models/person_vs_background_vs_random_pre_trained_alex_net/person_vs_background_vs_random_alex_net_pre_trained_lr_0.001_iter_40000.solverstate
