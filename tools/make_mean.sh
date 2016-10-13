#!/usr/bin/env sh
# Compute the mean image from the person training lmdb
# N.B. this is available in ../data/

DATA=data/lmdb
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/compute_image_mean $DATA/pascal_plus_21cls_train_lmdb \
  $DATA/21class_mean.binaryproto

echo "Done."
