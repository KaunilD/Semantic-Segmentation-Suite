#!/bin/bash -e

#source activate tensorflow

model=PSPNet-Res50

python main.py --mode train --dataset HT --num_val_images 498 --h_flip True --model $model | tee $model-Train.txt
python main.py --mode val --dataset HT --model $model 2>&1 | tee $model-Val.txt
python main.py --mode test --dataset HT --model $model 2>&1 | tee $model-Test.txt
