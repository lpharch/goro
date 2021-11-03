import os
import pandas as pd
import shutil
import argparse


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

all_mixes = os.listdir(simpts)

os.system("tsp -S 44")
all_mixes=["mix128"]

for round in range(1000):
    cmd = ""
    cmd += (gem5 + "/build/ARM/gem5.opt ")
    cmd += ("-d " + results + "/mix128_" + str(round) + " ")
    cmd += (gem5 + "/configs/example/fs.py ")
    cmd += ("--caches ")
    cmd += ("--kernel /home/cc/disks/binaries/vmlinux.arm64 ")
    cmd += ("--disk-image /home/cc/disks/disks/ubuntu-18.04-arm64-docker_big.img ")
    cmd += ("--cpu-type=DerivO3CPU ")
    cmd += ("--restore-simpoint-checkpoint -r 1 ")
    cmd += ("--checkpoint-dir " + simpts + "/mix128 ")
    cmd += ("--restore-with-cpu=AtomicSimpleCPU ")
    cmd += ("--l3cache --l3-hwp-type=L3MultiPrefetcher ")
    cmd += ("--l2-hwp-type=L2MultiPrefetcher ")
    cmd += ("--l1d-hwp-type=L1MultiPrefetcher ")
    cmd += ("--mem-size=64GB --mem-type=DDR4_2400_8x8 ")
    cmd += ("-n 4 ")
    cmd += ("--mode Real ")
    cmd += ("--real ")
    cmd += ("--sample_length 500000 ")
    cmd += ("--num_sample 200 ")
    cmd += ("--binspath /home/cc/bins/bins_levels_32.bins ")
    cmd += ("--app mix128."+str(round)+" ")
    cmd += (" > " + output + "/mix128."+str(round)+".out ")
    
    st = 'tsp bash -c "' + cmd+'"'
    print(st)    
    os.system(st)
