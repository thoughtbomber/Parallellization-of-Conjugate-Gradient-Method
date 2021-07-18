#!/usr/bin/bash -l
#   SBATCH -N 10  
#SBATCH -n 4
#  SBATCH --reservation phpc2021
#  SBATCH --account phpc2021


# We are using only one node do to shared memory parallelism
# We use only one task (process)
# and n cpus_per_task which are the maximum number of threads that we can use

# if you compiled with intel comment the gcc line and uncomment the intel one
# module load intel

module load gcc openblas
module load gcc mvapich2
salloc
#  srun ./cgsolver 1138_bus.mtx
# srun ./cgsolver matrix_create.mtx
srun ./cgsolver lap2D_5pt_n100.mtx
#  srun ./cgsolver poisson_16384.mtx