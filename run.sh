#!/bin/bash -e

input_dir=$1
output_dir=$2
params=$3

model_name=$(echo $params | json model-name)

python main.py --mode train --dataset HT --num_val_images 498 --h_flip True --model ${model_name} | tee ${model_name}-Train.txt
python main.py --mode val --dataset HT --model ${model_name} 2>&1 | tee ${model_name}-Val.txt
python main.py --mode test --dataset HT --model ${model_name} 2>&1 | tee ${model_name}-Test.txt
