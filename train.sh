#!/usr/bin/env bash

if [ $# != 2 ];then
  echo "Usage:$0 stage speaker"
  exit 1
fi

stage=$1
speaker=$2

# extract acoustic feature (without dynamic features)
if [ $stage -le 1 ];then
  python local/extract_acoustic_feat.py --br bin/ --dr data/$speaker
fi

# copy synthesis to verify extracted acoustic features
if [ $stage -le 2 ];then
  python local/copy_syn.py --br bin/ --dr data/$speaker
fi

# preprocess acoustic feature and text(lingual features)
if [ $stage -le 3 ];then
  python preprocess.py
fi

# train model
if [ $stage -le 4 ];then
  python train.py
fi

# synthesis
if [ $stage -le 5 ];then

fi