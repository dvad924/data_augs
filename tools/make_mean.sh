#!/usr/bin/env sh
# Compute the mean image from the person training lmdb
# N.B. this is available in ../data/

DATA=data/person_only_lmdb
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/compute_image_mean $DATA/person_vs_background_vs_random_256_train_lmdb \
  $DATA/coco_color_mean.binaryproto

echo "Done."
