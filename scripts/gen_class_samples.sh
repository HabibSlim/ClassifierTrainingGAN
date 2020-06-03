#!/bin/bash
export PATH="/home/mrim/quenot/anaconda3/bin:$PATH"
echo "Generating class-conditional samples..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 46k_it \
--ofile sample_test \
--num_samples 6 --class 9
