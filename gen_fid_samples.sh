#!/bin/bash
export PATH="/home/mrim/quenot/anaconda3/bin:$PATH"
echo "Generating FID samples..."

echo "55k_hier_z..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 55k_hier_z \
--ofile 55k_hier_z \
--num_samples 5000
echo "Done."
echo " "

