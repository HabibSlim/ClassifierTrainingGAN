#!/bin/bash
export PATH="/home/mrim/quenot/anaconda3/bin:$PATH"
echo "Generating samples..."
CUDA_VISIBLE_DEVICES=0 python sample.py \
--model 55k_hier_z \
--ofile 55k_samples \
--num_samples 1000 \
--truncate 0.5
