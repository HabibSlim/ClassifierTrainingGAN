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


echo "55k_hier_z_trunc15..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 55k_hier_z \
--ofile 55k_hier_z_trunc15 \
--num_samples 5000 \
--truncate 1.5
echo "Done."
echo " "

echo "55k_hier_z_trunc05..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 55k_hier_z \
--ofile 55k_hier_z_trunc05 \
--num_samples 5000 \
--truncate 0.5
echo "Done."
echo " "


echo "hard_norm_mu_it_1..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'hard_norm_mu/gan_multi_1' \
--ofile hard_norm_mu_it_1 \
--num_samples 5000
echo "Done."
echo " "

echo "hard_norm_mu_it_2..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'hard_norm_mu/gan_multi_2' \
--ofile hard_norm_mu_it_2 \
--num_samples 5000
echo "Done."
echo " "


echo "hard_norm_std_it_1..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'hard_norm_std/gan_multi_1' \
--ofile hard_norm_std_it_1 \
--num_samples 5000
echo "Done."
echo " "

echo "hard_norm_std_it_2..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'hard_norm_std/gan_multi_2' \
--ofile hard_norm_std_it_2 \
--num_samples 5000
echo "Done."
echo " "


echo "soft_norm_it_1..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'soft_norm/gan_multi_1' \
--ofile soft_norm_it_1 \
--num_samples 5000
echo "Done."
echo " "

echo "soft_norm_it_2..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'soft_norm/gan_multi_2' \
--ofile soft_norm_it_2 \
--num_samples 5000
echo "Done."
echo " "


echo "hard_norm_mu_multi_n2..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'hard_norm_mu/gan_multi' \
--ofile hard_norm_mu_multi_n2 \
--num_samples 5000 \
--multi_gans 2
echo "Done."
echo " "


echo "hard_norm_std_multi_n2..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'hard_norm_std/gan_multi' \
--ofile hard_norm_std_multi_n2 \
--num_samples 5000 \
--multi_gans 2
echo "Done."
echo " "


echo "soft_norm_multi_n2..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'soft_norm/gan_multi' \
--ofile soft_norm_multi_n2 \
--num_samples 5000 \
--multi_gans 2
echo "Done."
echo " "


echo "gan_multi_n2..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'multi_random/gan_multi' \
--ofile multi_random_n2 \
--num_samples 5000 \
--multi_gans 2
echo "Done."
echo " "


echo "gan_multi_n3..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'multi_random/gan_multi' \
--ofile multi_random_n3 \
--num_samples 5000 \
--multi_gans 3
echo "Done."
echo " "


echo "gan_multi_n4..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'multi_random/gan_multi' \
--ofile multi_random_n4 \
--num_samples 5000 \
--multi_gans 4
echo "Done."
echo " "


echo "gan_multi_n5..."
CUDA_VISIBLE_DEVICES=0 python sample_class.py \
--model 'multi_random/gan_multi' \
--ofile multi_random_n5 \
--num_samples 5000 \
--multi_gans 5
echo "Done."
echo " "
