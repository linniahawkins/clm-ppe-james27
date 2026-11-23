#!/bin/bash
#PBS -N hmatch_num
#PBS -q casper
#PBS -l walltime=6:00:00
#PBS -A P93300041
#PBS -j oe
#PBS -k eod
#PBS -l select=1:ncpus=1

source ~/.bashrc
conda activate /glade/work/linnia/conda-envs/mlenv

python hmatch_check.py num

