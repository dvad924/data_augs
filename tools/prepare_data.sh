#!/usr/bin/env sh
#create the peta lmdb inputs
# N.B. set the path to the peta train + val data dirs

EXAMPLE=data/lmdb
DATA=data
METADATA=data/person/assign
TOOLS="$CAFFE_ROOT/build/tools"

TRAIN_DATA_ROOT="$DATA/person/images/"
TEST_DATA_ROOT="$DATA/person/images/"

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
		$METADATA/trainclas.txt \
		$EXAMPLE/people_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
		--resize_height=$RESIZE_HEIGHT \
		--resize_width=$RESIZE_WIDTH \
		--shuffle \
		$TEST_DATA_ROOT \
		$METADATA/testclas.txt \
		$EXAMPLE/people_test_lmdb

echo "Done."
