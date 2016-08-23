#!/usr/bin/env sh
#create the peta lmdb inputs
# N.B. set the path to the peta train + val data dirs
COCODATA=/home/dl/DATA/coco
EXAMPLE=data/lmdb
DATA=data
METADATA="$COCODATA/"
TOOLS="$CAFFE_ROOT/build/tools"

TRAIN_DATA_ROOT="$COCODATA/image_patches/"
TEST_DATA_ROOT="$COCODATA/image_patches/"

RESIZE=true
if $RESIZE; then
    RESIZE_HEIGHT=256
    RESIZE_WIDTH=256
else
    RESIZE_HEIGHT=0
    RESIZE_HEIGHT=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_peta.sh to the path" \
       "where the peta training data is stored."
  exit 1
fi


if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: TEST_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the TEST_DATA_ROOT variable in create_peta.sh to the path" \
       "where the peta validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."
echo "$TRAIN_DATA_ROOT"
echo "$TEST_DATA_ROOT"
GLOG_logtostderr=1 $TOOLS/convert_imageset \
		--resize_height=$RESIZE_HEIGHT \
		--resize_width=$RESIZE_WIDTH \
		--shuffle \
		$TRAIN_DATA_ROOT \
		$METADATA/train.txt \
		$EXAMPLE/coco256_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
		--resize_height=$RESIZE_HEIGHT \
		--resize_width=$RESIZE_WIDTH \
		--shuffle \
		$TEST_DATA_ROOT \
		$METADATA/val.txt \
		$EXAMPLE/coco256_val_lmdb

echo "Done."
