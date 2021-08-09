# Written by Majid
# creating mixes to run
import random

num_cores = 4
num_mixes = 15

spec_path: str = "/home/gem5/spec2017/benchspec/CPU/"
run = "/run/run_base_refrate_mytest-64.0000/"
apps = ["500.perlbench_r", "503.bwaves_r", "507.cactuBSSN_r", "510.parest_r", "519.lbm_r",
        "521.wrf_r", "525.x264_r", "527.cam4_r", "538.imagick_r", "544.nab_r", "549.fotonik3d_r", "557.xz_r",
        "505.mcf_r", "508.namd_r", "511.povray_r", "520.omnetpp_r", "523.xalancbmk_r", "526.blender_r",
        "531.deepsjeng_r", "541.leela_r", "548.exchange2_r", "554.roms_r"]

run_cmds = {
    "500.perlbench_r": "./perlbench_r_base.mytest-64 -I./lib checkspam.pl 2500 5 25 11 150 1 1 1 1",
    "503.bwaves_r": "./bwaves_r_base.mytest-64 bwaves_1 < bwaves_1.in",
    "505.mcf_r": "./mcf_r_base.mytest-64 inp.in",
    "507.cactuBSSN_r": "./cactusBSSN_r_base.mytest-64 spec_ref.par",
    "508.namd_r": "./namd_r_base.mytest-64 --input apoa1.input --output apoa1.ref.output --iterations 65",
    "510.parest_r": "./parest_r_base.mytest-64 ref.prm",
    "511.povray_r": "./povray_r_base.mytest-64 SPEC-benchmark-ref.ini",
    "519.lbm_r": "./lbm_r_base.mytest-64 3000 reference.dat 0 0 100_100_130_ldc.of",
    "520.omnetpp_r": "./omnetpp_r_base.mytest-64 -c General -r 0",
    "521.wrf_r": "./wrf_r_base.mytest-64",
    "523.xalancbmk_r": "./cpuxalan_r_base.mytest-64 -v t5.xml xalanc.xsl",
    "525.x264_r": "./x264_r_base.mytest-64 --pass 1 --stats x264_stats.log --bitrate 1000 --frames 1000 -o BuckBunny_New.264 BuckBunny.yuv 1280x720",
    "526.blender_r": "./blender_r_base.mytest-64 sh3_no_char.blend --render-output sh3_no_char_ --threads 1 -b -F RAWTGA -s 849 -e 849 -a",
    "527.cam4_r": "./cam4_r_base.mytest-64",
    "531.deepsjeng_r": "./deepsjeng_r_base.mytest-64 ref.txt",
    "538.imagick_r": "./imagick_r_base.mytest-64 -limit disk 0 refrate_input.tga -edge 41 -resample 181% -emboss 31 -colorspace YUV -mean-shift 19x19+15% -resize 30% refrate_output.tga",
    "541.leela_r": "./leela_r_base.mytest-64 ref.sgf ",
    "544.nab_r": "./nab_r_base.mytest-64 1am0 1122214447 122",
    "548.exchange2_r": "./exchange2_r_base.mytest-64 6",
    "549.fotonik3d_r": "./fotonik3d_r_base.mytest-64",
    "554.roms_r": "./roms_r_base.mytest-64 < ocean_benchmark2.in.x",
    "557.xz_r": "./xz_r_base.mytest-64 cld.tar.xz 160 19cf30ae51eddcbefda78dd06014b4b96281456e078ca7c13e1c0c9e6aaea8dff3efb4ad6b0456697718cede6bd5454852652806a657bb56e07d61128434b474 59796407 61004416 6"
}

mix_num=0

for j in range(num_mixes):
    entry_list = list(run_cmds.items())
    app_num=0
    cmd =""
    f = open("mix" + str(mix_num) + ".rcS", "a")
    for i in range(num_cores):

        random_entry = random.choice(entry_list)
        entry_list.remove(random_entry)


        print("cd "+spec_path+random_entry[0]+run)
        f.write("cd "+spec_path+random_entry[0]+run+"\n")

        if app_num < num_cores - 1:
            print("taskset -c "+str(app_num)+" "+random_entry[1]+" &")
            f.write("taskset -c "+str(app_num)+" "+random_entry[1]+" &"+"\n")
        else:
            app_num = 0
            print("taskset -c "+str(num_cores-1)+" "+random_entry[1])
            f.write("taskset -c "+str(num_cores-1)+" "+random_entry[1]+"\n")
        app_num += 1
    f.close()

    mix_num += 1
    print("-----------")
