#!/bin/sh

export PYTHONPATH="../":"${PYTHONPATH}"
dataset=$1
model_path=$4
epoch=$6
seed=$7
method=$5
batch_size=$2
scoring=$3

DATA_DIR=/media/disk1/intern1001
#MODEL=$model_path-v$method-$epoch-$seed
MODEL=$model_path

python model_evaluate.py \
--dataset $dataset \
--model_path $MODEL \
--scoring $scoring \
--test_data_path $DATA_DIR \
--batch_size $batch_size \
