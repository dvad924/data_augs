#!/usr/bin/env sh
# Compute the mean image from the person training lmdb
# N.B. this is available in ../data/

DATA=data/lmdb
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/compute_image_mean $DATA/coco_train_lmdb \
  $DATA/coco_color_mean.binaryproto

echo "Done."
