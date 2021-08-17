import os
import pandas as pd
import shutil

num_mixes = 4
mixes_per_node = 2

simpts = "/work/05330/jalili/stampede2/goro/simpoints/ch0"
gem5 = "/work/05330/jalili/stampede2/goro/gem5"
scripts = "/work/05330/jalili/stampede2/goro/script"
points = "/work/05330/jalili/stampede2/goro/points"
results = "/work/05330/jalili/stampede2/goro/results"
output = "/work/05330/jalili/stampede2/goro/outputs"

simulation="baseline"

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

for chunk in range(chunk_counts):
    print("chunk", chunk)
    ch_name = "chunk_" + str(chunk)
    f = open(ch_name + ".sh", "w")
    f.write(template_dict["start"] + "\n")
    f.write(template_dict["jobName"] + ch_name + "\n")
    f.write(template_dict["out"] + "Out" + ch_name + ".%j.out" + "\n")
    f.write(template_dict["err"] + "Err" + ch_name + ".%j.err" + "\n")
    f.write(template_dict["queue"] + "skx-normal" + "\n")
    f.write(template_dict["tot_node"] + "1" + "\n")
    f.write(template_dict["mpi"] + "1" + "\n")
    f.write(template_dict["time"] + "03:00:00" + "\n")
    f.write(template_dict["mail_type"] + "\n")
    f.write(template_dict["allocation"] + "\n")
    f.write(template_dict["my_mail"] + "\n\n\n")
    f.write(template_dict["python3"] + "\n")
    f.write(template_dict["python3"] + "\n\n\n")
    f.write(template_dict["venv"] + "\n\n\n")
    for mix in range(mixes_per_node):
        app = all_mixes[app_idx]
        f.write(gem5 + "/build/ARM/gem5.opt ")
        f.write("-d " + results + "/" + app + " ")
        f.write(gem5 + "/configs/example/fs.py ")
        f.write("--caches ")
        f.write("--kernel /work/05330/jalili/stampede2/goro/disks/binaries/vmlinux.arm64 ")
        f.write("--disk-image /work/05330/jalili/stampede2/goro/disks/disks/ubuntu-18.04-arm64-docker_big.img ")
        f.write("--cpu-type=DerivO3CPU ")
        f.write("--restore-simpoint-checkpoint -r 1 ")
        f.write("--checkpoint-dir " + simpts + "/" + app + " ")
        f.write("--restore-with-cpu=AtomicSimpleCPU ")
        f.write("--l3cache --l3-hwp-type=L3MultiPrefetcher ")
        f.write("--l2-hwp-type=L2MultiPrefetcher ")
        f.write("--l1d-hwp-type=L1MultiPrefetcher ")
        f.write("--mem-size=64GB --mem-type=DDR4_2400_8x8 ")
        f.write("-n 4 ")
        f.write("--mode "+ simulation)
        f.write(" > " + output + "/" + app + ".out")
        if (mix < mixes_per_node - 1):
            f.write(" & ")
        f.write("\n")
        app_idx += 1
    f.write("wait")	
    f.close()
    os.system("sbatch ./" + ch_name + ".sh")
