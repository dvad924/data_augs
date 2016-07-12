#!/usr/bin/env sh
# Compute the mean image from the person training lmdb
# N.B. this is available in ../data/

DATA=data/person_only_lmdb
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/compute_image_mean $DATA/people_patch_train_lmdb \
  $DATA/person_patch_color_mean.binaryproto

echo "Done."
