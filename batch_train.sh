#!/bin/bash

data=$1
output=$2
nfolds=$3
start=$4
end=$5
smooth_value=$6
smooth_stats=$7
resume=$8
gpu=$9


if [[ -n "$data" ]] && [[ -n "$start" ]] && [[ -n "$end" ]]; then
    for i in $(eval echo {$start..$end})
    do
      CUDA_VISIBLE_DEVICES=$gpu python train.py --data_dir=$data --output_dir=$output --n_folds=$nfolds --train_epochs=100 --fold_idx=$i --smooth_value=$smooth_value --smooth_stats=$smooth_stats --resume=$resume
    done

else
    echo "argument error"
fi

