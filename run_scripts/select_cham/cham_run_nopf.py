import os
import pandas as pd
import shutil

num_mixes = 4
mixes_per_node = 2

simpts = "/home/cc/mixes"
gem5 = "/home/cc/goro/gem5"
results = "/home/cc/goro/results/"
output = "/home/cc/goro/outputs/"

simulation="nopf"

dir = results+simulation
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
results += simulation

dir = output+simulation
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
output += simulation

template_dict = {
    "start": "#!/bin/bash",
    "jobName": "#SBATCH -J ",
    "out": "#SBATCH -o ",
    "err": "#SBATCH -e ",
    "queue": "#SBATCH -p ",
    "tot_node": "#SBATCH -N ",
    "mpi": "#SBATCH -n ",
    "time": "#SBATCH -t ",
    "mail_type": "#SBATCH --mail-type=all",
    "allocation": "#SBATCH -A lph_mem ",
    "my_mail": "#SBATCH --mail-user=majid@utexas.edu",
    "python3": "module load python3",
    "venv": "source /work/05330/jalili/stampede2/goro/gem5/v21/bin/activate",

}

all_mixes = os.listdir(simpts)
chunk_counts = int(len(all_mixes) / mixes_per_node)
app_idx = 0
# ts -S 3
os.system("tsp -S 44")

for app in (all_mixes):
    cmd = ""
    cmd += (gem5 + "/build/ARM/gem5.opt ")
    cmd += ("-d " + results + "/" + app + " ")
    cmd += (gem5 + "/configs/example/fs.py ")
    cmd += ("--caches ")
    cmd += ("--kernel /home/cc/goro/disks/binaries/vmlinux.arm64 ")
    cmd += ("--disk-image /home/cc/goro/disks/disks/ubuntu-18.04-arm64-docker_big.img ")
    cmd += ("--cpu-type=DerivO3CPU ")
    cmd += ("--restore-simpoint-checkpoint -r 1 ")
    cmd += ("--checkpoint-dir " + simpts + "/" + app + " ")
    cmd += ("--restore-with-cpu=AtomicSimpleCPU ")
    cmd += ("--l3cache --l3-hwp-type=L3MultiPrefetcher ")
    cmd += ("--l2-hwp-type=L2MultiPrefetcher ")
    cmd += ("--l1d-hwp-type=L1MultiPrefetcher ")
    cmd += ("--mem-size=64GB --mem-type=DDR4_2400_8x8 ")
    cmd += ("-n 4 ")
    cmd += ("--mode "+ simulation)
    cmd += (" > " + output + "/" + app + ".out")

    
    
    print("cmd", cmd)    
    os.system("tsp " + cmd)
