#!/bin/bash

priors=( gaussian uniform laplace student-t cauchy improper gaussian_gamma gaussian_uniform horseshoe )
datasets=( boston wine energy naval concrete kin8nm power yacht protein )
inference=( SGLD )  # add HMC if needed
scales=( 0.14 1.41 14.1 )
temps=( 0.0 0.1 1.0 )

for prior in "${priors[@]}"
do
    for scale in "${scales[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for inf in "${inference[@]}"
            do
                for temp in "${temps[@]}"
                do
                    bsub -n 2 -W 4:00 -J "bnn" -sp 40 -g /vincent/experiments -G ms_raets -R "rusage[mem=4000,ngpus_excl_p=1]" "source activate bnn; python train_bnn.py with weight_prior=$prior data=UCI_$dataset inference=$inf warmup=5000 burnin=1000 weight_scale=$scale cycles=20 n_samples=100 skip=100 temperature=$temp"
                done
            done
        done
    done
done
