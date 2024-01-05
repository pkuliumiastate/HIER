#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1-00:00:00  # max job runtime
#SBATCH --cpus-per-task=8  # number of processor cores
#SBATCH --nodes=1  # number of nodes
#SBATCH --partition=gpu  # partition(s)
#SBATCH --gres=gpu:1
#SBATCH --mem=40G  # max memory


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load ml-gpu/20220928
cd /work/LAS/wzhang-lab/mingl/HIER
ml-gpu /work/LAS/wzhang-lab/mlgpuvenv-20220928/bin/python Hier-Local-QSGD.py --dataset cifar10 --model cnn_complex --num_clients 100 --num_honest_client 100 --alpha 1 --attack target_attack 
