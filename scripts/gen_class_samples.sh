#!/bin/bash
export PATH="/home/mrim/quenot/anaconda3/bin:$PATH"
echo "Generating class-conditional samples..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 55k_hier_z \
--ofile gen_samples \
--num_samples 5000 \
--torch_format
