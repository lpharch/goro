import os
import pandas as pd
import shutil
import argparse


num_mixes = 4
mixes_per_node = 2

simpts = "/home/cc/mixes"
gem5 = "/home/cc/goro/gem5"
results = "/home/cc/goro/results/"
output = "/home/cc/goro/outputs/"


parser = argparse.ArgumentParser('parameters')
parser.add_argument('--simulation', type=str, default=True, help="(default: True)")

args = parser.parse_args()
simulation = args.simulation



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

intravl_lengh=[10000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]

all_mixes = os.listdir(simpts)
chunk_counts = int(len(all_mixes) / mixes_per_node)
app_idx = 0
# ts -S 3
os.system("tsp -S 60")

for length in intravl_lengh:
    for app in (all_mixes):
        cmd = ""
        cmd += (gem5 + "/build/ARM/gem5.opt ")
        cmd += ("-d " + results + "/" + app+".length"+str(length) + " ")
        cmd += (gem5 + "/configs/example/fs.py ")
        cmd += ("--caches ")
        cmd += ("--kernel /home/cc/disks/binaries/vmlinux.arm64 ")
        cmd += ("--disk-image /home/cc/disks/disks/ubuntu-18.04-arm64-docker_big.img ")
        cmd += ("--cpu-type=DerivO3CPU  --bp-typ=TAGE ")
        cmd += ("--restore-simpoint-checkpoint -r 1 ")
        cmd += ("--checkpoint-dir " + simpts + "/" + app + " ")
        cmd += ("--restore-with-cpu=AtomicSimpleCPU ")
        cmd += ("--l3cache --l3-hwp-type=L3MultiPrefetcher ")
        cmd += ("--l2-hwp-type=L2MultiPrefetcher ")
        cmd += ("--l1d-hwp-type=L1MultiPrefetcher ")
        cmd += ("--mem-size=64GB --mem-type=DDR4_2400_16x4 ")
        cmd += ("-n 4 ")
        cmd += ("--inference ")
        cmd += ("--sample_length "+str(length)+" ")
        cmd += ("--num_sample "+str(int(100000000/length))+" ")
        cmd += ("--model /home/cc/model_7 ")
        cmd += ("--mode random ")
        cmd += (" > " + output + "/" + app +".length"+str(length)+".out")
        st = 'tsp bash -c "' + cmd+'"'
        print(st)    
        os.system(st)
