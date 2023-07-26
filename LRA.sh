#!/bin/bash

#SBATCH -J relearn
#SBATCH -e /home/dl68vicy/logfiles_cluster/stderr.relearn.%j.txt
#SBATCH -o /home/dl68vicy/logfiles_cluster/stdout.relearn.%j.txt
#SBATCH -n 1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=1-0
#SBATCH --cpus-per-task=12
#SBATCH --account=project02091

module restore

backgroundactivity=0.003
initialfired=0.0

beta=0.001 #default is ok
calciumdecay=10000

mincalcaxons=0.1 #default 0.4
mincalcdends=0.025 #default 0.1

growthrateaxons=0.001
growthratedendrites=0.001
growthratedecay=1.0

steps=400000

gausssigma=750

logpath=/home/dl68vicy/sim_7_07_400k_LRA

mkdir -p $logpath

srun /home/dl68vicy/relearn/relearn/build/bin/relearn \
-n 50000 \
--log-path $logpath \
--openmp 12 \
--algorithm low-rank-approx \
--low_rank_algo centroid \
--rank 3 \
--steps $steps \
--kernel-type gaussian \
--calcium-log-step 5000 \
--synaptic-elements-lower-bound 0.7 \
--synaptic-elements-upper-bound 1.2 \
--growth-rate-axon $growthrateaxons \
--growth-rate-dendrite $growthratedendrites \
--background-activity constant \
--base-background-activity $backgroundactivity \
