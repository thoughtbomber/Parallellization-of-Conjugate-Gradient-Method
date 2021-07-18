#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#  SBATCH --account=phpc2021
#  SBATCH --reservation=phpc2021

module load gcc cuda

salloc
srun ./cgsolver lap2D_5pt_n100.mtx 20

