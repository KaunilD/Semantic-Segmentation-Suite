#!/bin/bash -ex

input_dir=$1
output_dir=$2
params=$3

dataset_name=$(echo $params | json dataset-name)
./prep-data.sh ${dataset_name} ${input_dir}

model_name=$(echo $params | json model-name)
python main.py --mode train --dataset ${dataset_name} --num_val_images 498 --h_flip True --model ${model_name} | tee ${model_name}-Train.txt
python main.py --mode val --dataset ${dataset_name} --model ${model_name} 2>&1 | tee ${model_name}-Val.txt
python main.py --mode test --dataset ${dataset_name} --model ${model_name} 2>&1 | tee ${model_name}-Test.txt
