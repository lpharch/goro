import os
import pandas as pd
import shutil

num_mixes = 4
mixes_per_node = 2

simpts = "/home/cc/goro/simpoints/allmixes"
gem5 = "/home/cc/goro/gem5"
scripts = "/home/cc/goro/script"
points = "/home/cc/goro/points"
results = "/home/cc/goro/results/"
output = "/home/cc/goro/outputs/"

simulation="dataset"

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
os.system("ts -S 40")


degrees = [
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0",
            "0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0",
            "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1",
            "0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,1,1",
        
          ]

for i, d in enumerate(degrees):
    for app in (all_mixes):
        cmd = ""
        cmd += (gem5 + "/build/ARM/gem5.opt ")
        cmd += ("-d " + results + "/" + app + "_low_d"+str(i)+" ")
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
        cmd += ("--mode custom ")
        cmd += ("--train ")
        cmd += ("--degrees "+d+" ")
        cmd += ("--app "+app+".low_d"+str(i)+" ")

        
        
        print("cmd", cmd)    
        os.system("tsp " + cmd)
