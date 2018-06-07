#!/bin/bash -e

input_dir=$1
output_dir=$2
params=$3

# Prep the data.
dataset_name=$(echo $params | json dataset-name)
cd ${dataset_name}
sed -i "s,INPUT_DIR,$input_dir,g" train.txt
sed -i "s,INPUT_DIR,$input_dir,g" val.txt
sed -i "s,INPUT_DIR,$input_dir,g" test.txt
python prep_data.py
cd ..

# Train the model.
model_name=$(echo $params | json model-name)
python main.py --mode train --dataset ${dataset_name} --num_val_images 498 --h_flip True --model ${model_name} | tee ${model_name}-Train.txt
python main.py --mode val --dataset ${dataset_name} --model ${model_name} 2>&1 | tee ${model_name}-Val.txt
python main.py --mode test --dataset ${dataset_name} --model ${model_name} 2>&1 | tee ${model_name}-Test.txt
