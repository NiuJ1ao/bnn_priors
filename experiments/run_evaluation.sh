#!/bin/bash

exp_dir="/data2/users/yn621/cold-posterior-cnn/results/exp_cifar_scale0"
calibration_data="cifar10c-gaussian_blur"
ood_data="svhn"
skip=50
device=3
kernel_size=3
depth=20

config_files=($(ls $exp_dir/temp1.0/config.json))

for conf_file in ${config_files[@]}
do
    CUDA_VISIBLE_DEVICES=$device python eval_bnn.py with width=$kernel_size depth=$depth config_file=$conf_file skip_first=$skip
    # python eval_bnn.py with config_file=$conf_file eval_data=$calibration_data calibration_eval=True skip_first=$skip
    # CUDA_VISIBLE_DEVICES=$device python eval_bnn.py with width=$kernel_size depth=$depth config_file=$conf_file eval_data=$ood_data ood_eval=True skip_first=$skip
done
