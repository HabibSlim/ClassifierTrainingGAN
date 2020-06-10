#!/bin/bash
N_GANS=2

#export PATH="/home/mrim/quenot/anaconda3/bin:$PATH"
export PATH="/srv/storage/irim@storage1.lille.grid5000.fr/anaconda3/bin:$PATH"

echo "Running SigmoidNorm it=1 n=" $N_GANS "(it+1)..."
CUDA_VISIBLE_DEVICES=0 python train_classifier.py \
--num_batches 40 \
--batch_size 125 \
--model 'sigmoid_norm/gan_multi' \
--ofile trained_net \
--threshold 0.9 \
--num_workers 1 \
--epochs 180 \
--transform \
--filter_samples \
--multi_gans $N_GANS

