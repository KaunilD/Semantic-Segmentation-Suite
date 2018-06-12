#!/bin/bash -e

input_dir=$1
output_dir=$2
params=$3

dataset_name=$(echo $params | json dataset-name)
./prep-data.sh ${dataset_name} ${input_dir}

model_name=$(echo $params | json model-name)

function update_extra_args {
    set -x
    params=$1
    extra_args=$2
    keyword=$3
    echo $params | grep -q $keyword
    if [ $? -eq 0 ]
    then
        value=$(echo $params | json $keyword)
        command_line_arg=$(echo $keyword | tr '-' '_')
        extra_args="$extra_args --${command_line_arg} $value"
    fi
    echo $extra_args
}

extra_args=""
extra_args=$(update_extra_args "$params" "$extra_args" class-balancing)
extra_args=$(update_extra_args "$params" "$extra_args" loss-func)
extra_args=$(update_extra_args "$params" "$extra_args" learning-rate)

python main.py --mode train --dataset ${dataset_name} --num_val_images 498 --h_flip True --model ${model_name} $extra_args | tee ${model_name}-Train.txt
python main.py --mode val --dataset ${dataset_name} --model ${model_name} 2>&1 | tee ${model_name}-Val.txt
python main.py --mode test --dataset ${dataset_name} --model ${model_name} 2>&1 | tee ${model_name}-Test.txt
