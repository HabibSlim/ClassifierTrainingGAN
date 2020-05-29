#!/bin/bash
#export PATH="/home/mrim/quenot/anaconda3/bin:$PATH"
export PATH="/srv/storage/irim@storage1.lille.grid5000.fr/anaconda3/bin:$PATH"
echo "Running experiment B (Inf DSET)..."
CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
--num_batches 40 \
--batch_size 125 \
--model 55k_hier_z \
--ofile trained_net \
--threshold 0.9 \
--num_workers 1 \
--epochs 180
