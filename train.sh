#!/bin/sh

export PYTHONPATH="../":"${PYTHONPATH}"
dataset=$1 
train_type=$2
epoch=$3
batch_size=$4
scoring=$5
sampling_version=$6

#OUT=/home/donaldo9603/workspace/beir/examples/retrieval/training/output/$dataset/$train_type
OUT=/media/disk1/intern1001/output/$dataset/$train_type
#DATA_DIR=../../dataset
DATA_DIR=/media/disk1/intern1001

/home/intern1001/anaconda3/bin/python3 train_sbert_mod.py \
--dataset $dataset \
--train_type $train_type \
--epoch $epoch \
--output_dir $OUT \
--scoring $scoring \
--training_data_path $DATA_DIR \
--batch_size $batch_size \
--sampling_version $sampling_version
