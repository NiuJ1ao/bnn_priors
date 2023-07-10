#!/bin/bash

temp=1
scale_prior=0
device=0
cycles=60
warmup=45
depth=50
kernel_size=3
lr=0.01
data="cifar10"
log_dir="/data2/users/yn621/cold-posterior-cnn/results/exp_${data}_depth${depth}_width${kernel_size}_lr${lr}_warmup${warmup}_cycles${cycles}_scale${scale_prior}"

CUDA_VISIBLE_DEVICES=$device python train_bnn.py with data=$data model=googleresnet weight_prior=gaussian depth=$depth width=$kernel_size inference=VerletSGLDReject warmup=$warmup burnin=0 skip=1 n_samples=300 lr=$lr momentum=0.994 weight_scale=1.41 cycles=$cycles batch_size=128 temperature=$temp save_samples=True progressbar=True log_dir=$log_dir batchnorm=True scale_prior=$scale_prior