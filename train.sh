#!/usr/bin/env bash

if [ $# != 2 ];then
  echo "Usage:$0 stage speaker"
  exit 1
fi

stage=$1
dataset=$2

. path.sh

# preprocess acoustic feature and text(lingual features)
if [ $stage -le 0 ];then
  python preprocess.py --dataset=$dataset
fi

# train model
if [ $stage -le 1 ];then
  python train.py --base_dir=data/$dataset
fi

