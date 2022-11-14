#!/bin/sh

export PYTHONPATH="../":"${PYTHONPATH}"
dataset=fiqa
train_type=few
epoch=1
batch_size=16
scoring=cos_sim
sampling_version=1

OUT=/media/disk1/intern1001/output/$dataset/$train_type
DATA_DIR=/media/disk1/intern1001

EPOCHS="10"
SEEDS=$(seq 1001 1005)
VERSIONS=$(seq 2 4)

for e in $EPOCHS
do
	for v in $VERSIONS
	do
		for s in $SEEDS
		do
			/home/intern1001/anaconda3/bin/python3 train_sbert_mod.py \
			--dataset $dataset \
			--train_type $train_type \
			--epoch $e \
			--output_dir $OUT \
			--scoring $scoring \
			--training_data_path $DATA_DIR \
			--batch_size $batch_size \
			--sampling_version $v \
			--seed $s
		done
	done
done

/home/intern1001/anaconda3/bin/python3 train_sbert_mod.py \
                        --dataset $dataset \
                        --train_type $train_type \
                        --epoch 5 \
                        --output_dir $OUT \
                        --scoring $scoring \
                        --training_data_path $DATA_DIR \
                        --batch_size $batch_size \
                        --sampling_version 4 \
                        --seed 1005
