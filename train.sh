#!/usr/bin/env bash

if [ $# != 2 ];then
  echo "Usage:$0 stage speaker"
  exit 1
fi

stage=$1
speaker=$2

# extract acoustic feature (including dynamic features)
if [ $stage -le 1 ];then
  python local/extract_acoustic_feat.py --br bin/ --dr data/$speaker
fi

# copy synthesis
if [ $stage -le 2 ];then
  python local/copy_syn.py --br bin/ --dr data/$speaker
fi