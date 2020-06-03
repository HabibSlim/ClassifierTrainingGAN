#!/bin/bash
N_GANS=3

export PATH="/home/mrim/quenot/anaconda3/bin:$PATH"
#export PATH="/srv/storage/irim@storage1.lille.grid5000.fr/anaconda3/bin:$PATH"

echo "Running MultiGANs training, n=" $N_GANS "..."
CUDA_VISIBLE_DEVICES=0 python test.py \
--num_batches 40 \
--batch_size 125 \
--model gan_multi \
--ofile trained_net \
--threshold 0.9 \
--num_workers 1 \
--epochs 180 \
--transform \
--filter_samples \
--multi_gans $N_GANS