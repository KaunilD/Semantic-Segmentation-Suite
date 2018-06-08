#!/bin/bash

dataset_name=$1
input_dir=$2

BIN="$(dirname $(readlink -f -- "$0"))"
cd $BIN/${dataset_name}

sed -i "s,INPUT_DIR,$input_dir,g" train.txt
sed -i "s,INPUT_DIR,$input_dir,g" val.txt
sed -i "s,INPUT_DIR,$input_dir,g" test.txt
python prep_data.py
