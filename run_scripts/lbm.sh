#!/bin/bash
#SBATCH -J lbm             # Job name
#SBATCH -o lbmj.out        # Name of stdout output file
#SBATCH -e lbmj.err        # Name of stderr error file
#SBATCH -p skx-normal      # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 20:15:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A lph_mem         # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=majid@utexas.edu

# Any other commands must follow all #SBATCH directives...
module list
pwd
date
module load python3
source /work/05330/jalili/stampede2/RL_project/gem5/v21/bin/activate

simpts=/work/05330/jalili/stampede2/RL_project/simpoints
gem5=/work/05330/jalili/stampede2/RL_project/gem5
scripts=/work/05330/jalili/stampede2/RL_project/script
points=/work/05330/jalili/stampede2/RL_project/points


$gem5/build/ARM/gem5.opt -d $simpts/lbm $gem5/configs/example/fs.py --caches --kernel /work/05330/jalili/stampede2/RL_project/disks/binaries/vmlinux.arm64 --disk-image /work/05330/jalili/stampede2/RL_project/disks/disks/ubuntu-18.04-arm64-docker_big.img --mem-size=64GB --script $scripts/lbm.rcS -n 4 --take-simpoint-checkpoint=$points/619.lbm.simpoints,$points/619.lbm.weights,100000000,10000000 --checkpoint-dir=$gem5/m5out/ -r 1

