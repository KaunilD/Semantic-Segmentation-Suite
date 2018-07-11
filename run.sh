#!/bin/bash -e

input_dir=$1
output_dir=$2
params=$3

# Layout the data as the main script expects.
dataset_name=$(echo $params | json dataset-name)
if [ ! -d ${dataset_name}/train ]
then
    ./prep-data.sh ${dataset_name} ${input_dir}
fi

# Download checkpoints, if provided.
checkpoint_dir_uri=$(echo $params | json checkpoint-dir-uri)
if [[ ! -z $checkpoint_dir_uri ]]
then
    if [ ! -d checkpoints ]
    then
        aws s3 sync $checkpoint_dir_uri checkpoints
    fi
fi


###########################################################################
# If a keyword is present in the parameters, parse it out of the
# parameters and append its value as a command-line argument to
# the given (extra) command-line arguments.
###########################################################################
function update_extra_args {
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

# Parse optional arguments out of the parameters and turn them into
# command-line arguments for main.py.
extra_args=""
extra_args=$(update_extra_args "$params" "$extra_args" class-balancing)
extra_args=$(update_extra_args "$params" "$extra_args" loss-func)
extra_args=$(update_extra_args "$params" "$extra_args" learning-rate)
extra_args=$(update_extra_args "$params" "$extra_args" continue-training)
extra_args=$(update_extra_args "$params" "$extra_args" batch-size)
extra_args=$(update_extra_args "$params" "$extra_args" optimizer)

model_name=$(echo $params | json model-name)

# Train
python main.py --mode train \
    --dataset ${dataset_name} --model ${model_name} \
    --crop_height 360 --crop_width 480 --h_flip True $extra_args 2>&1 |
    tee ${model_name}-Train.txt

# Run the validation set with the best weights found during training.
python main.py --mode val --dataset ${dataset_name} --model ${model_name} \
    --crop_height 360 --crop_width 480 2>&1 | tee ${model_name}-Val.txt

# Run the test set with the best weights found during training.
python main.py --mode test --dataset ${dataset_name} --model ${model_name} \
    --crop_height 360 --crop_width 480 2>&1 | tee ${model_name}-Test.txt
