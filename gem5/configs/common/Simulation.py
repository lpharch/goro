# Copyright (c) 2012-2013 ARM Limited
# All rights reserved
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Copyright (c) 2006-2008 The Regents of The University of Michigan
# Copyright (c) 2010 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
from os import getcwd
from os.path import join as joinpath

from common import CpuConfig
from common import ObjectList

import m5
from m5.defines import buildEnv
from m5.objects import *
from m5.util import *

# Majid
from random import random
from random import randint
import pandas as pd
import torch
import numpy as np
from common import network

high_degree = True

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")
        

addToPath('../common')

def getCPUClass(cpu_type):
    """Returns the required cpu class and the mode of operation."""
    cls = ObjectList.cpu_list.get(cpu_type)
    return cls, cls.memory_mode()

def setCPUClass(options):
    """Returns two cpu classes and the initial mode of operation.

       Restoring from a checkpoint or fast forwarding through a benchmark
       can be done using one type of cpu, and then the actual
       simulation can be carried out using another type. This function
       returns these two types of cpus and the initial mode of operation
       depending on the options provided.
    """

    TmpClass, test_mem_mode = getCPUClass(options.cpu_type)
    CPUClass = None
    if TmpClass.require_caches() and \
            not options.caches and not options.ruby:
        fatal("%s must be used with caches" % options.cpu_type)

    if options.checkpoint_restore != None:
        if options.restore_with_cpu != options.cpu_type:
            CPUClass = TmpClass
            TmpClass, test_mem_mode = getCPUClass(options.restore_with_cpu)
    elif options.fast_forward:
        CPUClass = TmpClass
        TmpClass = AtomicSimpleCPU
        test_mem_mode = 'atomic'

    # Ruby only supports atomic accesses in noncaching mode
    if test_mem_mode == 'atomic' and options.ruby:
        warn("Memory mode will be changed to atomic_noncaching")
        test_mem_mode = 'atomic_noncaching'

    return (TmpClass, test_mem_mode, CPUClass)

def setMemClass(options):
    """Returns a memory controller class."""

    return ObjectList.mem_list.get(options.mem_type)

def setWorkCountOptions(system, options):
    if options.work_item_id != None:
        system.work_item_id = options.work_item_id
    if options.num_work_ids != None:
        system.num_work_ids = options.num_work_ids
    if options.work_begin_cpu_id_exit != None:
        system.work_begin_cpu_id_exit = options.work_begin_cpu_id_exit
    if options.work_end_exit_count != None:
        system.work_end_exit_count = options.work_end_exit_count
    if options.work_end_checkpoint_count != None:
        system.work_end_ckpt_count = options.work_end_checkpoint_count
    if options.work_begin_exit_count != None:
        system.work_begin_exit_count = options.work_begin_exit_count
    if options.work_begin_checkpoint_count != None:
        system.work_begin_ckpt_count = options.work_begin_checkpoint_count
    if options.work_cpus_checkpoint_count != None:
        system.work_cpus_ckpt_count = options.work_cpus_checkpoint_count

def findCptDir(options, cptdir, testsys):
    """Figures out the directory from which the checkpointed state is read.

    There are two different ways in which the directories holding checkpoints
    can be named --
    1. cpt.<benchmark name>.<instruction count when the checkpoint was taken>
    2. cpt.<some number, usually the tick value when the checkpoint was taken>

    This function parses through the options to figure out which one of the
    above should be used for selecting the checkpoint, and then figures out
    the appropriate directory.
    """

    from os.path import isdir, exists
    from os import listdir
    import re

    if not isdir(cptdir):
        fatal("checkpoint dir %s does not exist!", cptdir)

    cpt_starttick = 0
    if options.at_instruction or options.simpoint:
        inst = options.checkpoint_restore
        if options.simpoint:
            # assume workload 0 has the simpoint
            if testsys.cpu[0].workload[0].simpoint == 0:
                fatal('Unable to find simpoint')
            inst += int(testsys.cpu[0].workload[0].simpoint)

        checkpoint_dir = joinpath(cptdir, "cpt.%s.%s" % (options.bench, inst))
        if not exists(checkpoint_dir):
            fatal("Unable to find checkpoint directory %s", checkpoint_dir)

    elif options.restore_simpoint_checkpoint:
        # Restore from SimPoint checkpoints
        # Assumes that the checkpoint dir names are formatted as follows:
        dirs = listdir(cptdir)
        expr = re.compile('cpt\.simpoint_(\d+)_inst_(\d+)' +
                    '_weight_([\d\.e\-]+)_interval_(\d+)_warmup_(\d+)')
        cpts = []
        for dir in dirs:
            match = expr.match(dir)
            if match:
                cpts.append(dir)
        cpts.sort()

        cpt_num = options.checkpoint_restore
        if cpt_num > len(cpts):
            fatal('Checkpoint %d not found', cpt_num)
        checkpoint_dir = joinpath(cptdir, cpts[cpt_num - 1])
        match = expr.match(cpts[cpt_num - 1])
        if match:
            index = int(match.group(1))
            start_inst = int(match.group(2))
            weight_inst = float(match.group(3))
            interval_length = int(match.group(4))
            warmup_length = int(match.group(5))
        print("Resuming from", checkpoint_dir)
        simpoint_start_insts = []
        simpoint_start_insts.append(warmup_length)
        simpoint_start_insts.append(warmup_length + interval_length)
        testsys.cpu[0].simpoint_start_insts = simpoint_start_insts
        if testsys.switch_cpus != None:
            testsys.switch_cpus[0].simpoint_start_insts = simpoint_start_insts

        print("Resuming from SimPoint", end=' ')
        print("#%d, start_inst:%d, weight:%f, interval:%d, warmup:%d" %
            (index, start_inst, weight_inst, interval_length, warmup_length))

    else:
        dirs = listdir(cptdir)
        expr = re.compile('cpt\.([0-9]+)')
        cpts = []
        for dir in dirs:
            match = expr.match(dir)
            if match:
                cpts.append(match.group(1))

        cpts.sort(key = lambda a: int(a))

        cpt_num = options.checkpoint_restore
        if cpt_num > len(cpts):
            fatal('Checkpoint %d not found', cpt_num)

        cpt_starttick = int(cpts[cpt_num - 1])
        checkpoint_dir = joinpath(cptdir, "cpt.%s" % cpts[cpt_num - 1])

    return cpt_starttick, checkpoint_dir

def scriptCheckpoints(options, maxtick, cptdir):
    if options.at_instruction or options.simpoint:
        checkpoint_inst = int(options.take_checkpoints)

        # maintain correct offset if we restored from some instruction
        if options.checkpoint_restore != None:
            checkpoint_inst += options.checkpoint_restore

        print("Creating checkpoint at inst:%d" % (checkpoint_inst))
        exit_event = m5.simulate()
        exit_cause = exit_event.getCause()
        print("exit cause = %s" % exit_cause)

        # skip checkpoint instructions should they exist
        while exit_cause == "checkpoint":
            exit_event = m5.simulate()
            exit_cause = exit_event.getCause()

        if exit_cause == "a thread reached the max instruction count":
            m5.checkpoint(joinpath(cptdir, "cpt.%s.%d" % \
                    (options.bench, checkpoint_inst)))
            print("Checkpoint written.")

    else:
        when, period = options.take_checkpoints.split(",", 1)
        when = int(when)
        period = int(period)
        num_checkpoints = 0

        exit_event = m5.simulate(when - m5.curTick())
        exit_cause = exit_event.getCause()
        while exit_cause == "checkpoint":
            exit_event = m5.simulate(when - m5.curTick())
            exit_cause = exit_event.getCause()

        if exit_cause == "simulate() limit reached":
            m5.checkpoint(joinpath(cptdir, "cpt.%d"))
            num_checkpoints += 1

        sim_ticks = when
        max_checkpoints = options.max_checkpoints

        while num_checkpoints < max_checkpoints and \
                exit_cause == "simulate() limit reached":
            if (sim_ticks + period) > maxtick:
                exit_event = m5.simulate(maxtick - sim_ticks)
                exit_cause = exit_event.getCause()
                break
            else:
                exit_event = m5.simulate(period)
                exit_cause = exit_event.getCause()
                sim_ticks += period
                while exit_event.getCause() == "checkpoint":
                    exit_event = m5.simulate(sim_ticks - m5.curTick())
                if exit_event.getCause() == "simulate() limit reached":
                    m5.checkpoint(joinpath(cptdir, "cpt.%d"))
                    num_checkpoints += 1

    return exit_event

def benchCheckpoints(options, maxtick, cptdir):
    exit_event = m5.simulate(maxtick - m5.curTick())
    exit_cause = exit_event.getCause()

    num_checkpoints = 0
    max_checkpoints = options.max_checkpoints

    while exit_cause == "checkpoint":
        m5.checkpoint(joinpath(cptdir, "cpt.%d"))
        num_checkpoints += 1
        if num_checkpoints == max_checkpoints:
            exit_cause = "maximum %d checkpoints dropped" % max_checkpoints
            break

        exit_event = m5.simulate(maxtick - m5.curTick())
        exit_cause = exit_event.getCause()

    return exit_event

# Set up environment for taking SimPoint checkpoints
# Expecting SimPoint files generated by SimPoint 3.2
def parseSimpointAnalysisFile(options, testsys):
    import re

    simpoint_filename, weight_filename, interval_length, warmup_length = \
        options.take_simpoint_checkpoints.split(",", 3)
    print("simpoint analysis file:", simpoint_filename)
    print("simpoint weight file:", weight_filename)
    print("interval length:", interval_length)
    print("warmup length:", warmup_length)

    interval_length = int(interval_length)
    warmup_length = int(warmup_length)

    # Simpoint analysis output starts interval counts with 0.
    simpoints = []
    simpoint_start_insts = []

    # Read in SimPoint analysis files
    simpoint_file = open(simpoint_filename)
    weight_file = open(weight_filename)
    while True:
        line = simpoint_file.readline()
        if not line:
            break
        m = re.match("(\d+)\s+(\d+)", line)
        if m:
            interval = int(m.group(1))
        else:
            fatal('unrecognized line in simpoint file!')

        line = weight_file.readline()
        if not line:
            fatal('not enough lines in simpoint weight file!')
        m = re.match("([0-9\.e\-]+)\s+(\d+)", line)
        if m:
            weight = float(m.group(1))
        else:
            fatal('unrecognized line in simpoint weight file!')

        if (interval * interval_length - warmup_length > 0):
            starting_inst_count = \
                interval * interval_length - warmup_length
            actual_warmup_length = warmup_length
        else:
            # Not enough room for proper warmup
            # Just starting from the beginning
            starting_inst_count = 0
            actual_warmup_length = interval * interval_length

        simpoints.append((interval, weight, starting_inst_count,
            actual_warmup_length))

    # Sort SimPoints by starting inst count
    simpoints.sort(key=lambda obj: obj[2])
    for s in simpoints:
        interval, weight, starting_inst_count, actual_warmup_length = s
        print(str(interval), str(weight), starting_inst_count,
            actual_warmup_length)
        simpoint_start_insts.append(starting_inst_count)

    print("Total # of simpoints:", len(simpoints))
    testsys.cpu[0].simpoint_start_insts = simpoint_start_insts

    return (simpoints, interval_length)

def takeSimpointCheckpoints(simpoints, interval_length, cptdir):
    num_checkpoints = 0
    index = 0
    last_chkpnt_inst_count = -1
    for simpoint in simpoints:
        interval, weight, starting_inst_count, actual_warmup_length = simpoint
        if starting_inst_count == last_chkpnt_inst_count:
            # checkpoint starting point same as last time
            # (when warmup period longer than starting point)
            exit_cause = "simpoint starting point found"
            code = 0
        else:
            exit_event = m5.simulate()

            # skip checkpoint instructions should they exist
            while exit_event.getCause() == "checkpoint":
                print("Found 'checkpoint' exit event...ignoring...")
                exit_event = m5.simulate()

            exit_cause = exit_event.getCause()
            code = exit_event.getCode()

        if exit_cause == "simpoint starting point found":
            m5.checkpoint(joinpath(cptdir,
                "cpt.simpoint_%02d_inst_%d_weight_%f_interval_%d_warmup_%d"
                % (index, starting_inst_count, weight, interval_length,
                actual_warmup_length)))
            print("Checkpoint #%d written. start inst:%d weight:%f" %
                (num_checkpoints, starting_inst_count, weight))
            num_checkpoints += 1
            last_chkpnt_inst_count = starting_inst_count
        else:
            break
        index += 1

    print('Exiting @ tick %i because %s' % (m5.curTick(), exit_cause))
    print("%d checkpoints taken" % num_checkpoints)
    sys.exit(code)

def take_action(state, options):
    model_name = options.model
    state_space  = 65
    action_space = 19
    action_scale = 6
    acc = []
    # need to quantize the state first
    bins_file = open(options.binspath, 'r')
    Lines = bins_file.readlines()
    count = 0
    for line in Lines:
        line = line.replace('\n','')
        bins = line.split(" ")
        found_bin = False
        for b in range(1, len(bins)-1):
            if(state[count] >= float(bins[b]) and state[count] < float(bins[b+1]) ):
                found_bin = True
                state[count] = b
                break
        if(not found_bin):
            if(state[count] > state[len(bins)-1]):
                state[count] = len(bins) - 1
            else:
                state[count] = 0
        count += 1
    q_model = network.QNetwork(state_space, action_space, action_scale)
    checkpoint = torch.load((model_name), map_location=torch.device('cpu'))
    q_model.load_state_dict(checkpoint['modelA_state_dict'])
        
    out =  q_model(torch.tensor(state, dtype=torch.float))
    for tor in out:
        acc.append(torch.argmax(tor, dim=1)[[0]].item() )
    print("Actions suggested by the model ", acc)
    return acc
 
def read_state(testsys, np, app, timestamp):
    values = []
    keys = []
    
    for i in range(np):
        L1 = m5.getL1State(testsys, i)
        L1_keys = L1.keys()
        L1_values = L1.values()
        for k in L1_keys:
            keys.append("core"+str(i)+"."+k)
        for v in L1_values:
            values.append(v)
            
        L2 = m5.getL2State(testsys, i)
        L2_keys = L2.keys()
        L2_values = L2.values()
        for k in L2_keys:
            keys.append("core"+str(i)+"."+k)
        for v in L2_values:
            values.append(v)
     
    
    L3 =  m5.getL3State(testsys, 0)
    L3_keys = L3.keys()
    L3_values = L3.values()
    for k in L3_keys:
        keys.append("core"+str(i)+"."+k)
    for v in L3_values:
        values.append(v)
    df = pd.DataFrame(values, index=keys,  columns =[app+"_"+str(timestamp)])
    return df.T, values
        
def apply_degree(testsys, options, state):
    degrees = []
    np = options.num_cpus
    mode = options.mode
    # Order of results, I may make it dynamic
    components = ["Core0.L1.P0.degree", "Core0.L1.P1.degree",
                  "Core0.L2.P0.degree", "Core0.L2.P1.degree",
                  "Core1.L1.P0.degree", "Core1.L1.P1.degree",
                  "Core1.L2.P0.degree", "Core1.L2.P1.degree",
                  "Core2.L1.P0.degree", "Core2.L1.P1.degree",
                  "Core2.L2.P0.degree", "Core2.L2.P1.degree",
                  "Core3.L1.P0.degree", "Core3.L1.P1.degree",
                  "Core3.L2.P0.degree", "Core3.L2.P1.degree",
                  "LLC.P0.degree", "LLC.P1.degree", "LLC.P2.degree"   
                 ]
    L1_prefetcher_count = 2
    L2_prefetcher_count = 2
    L3_prefetcher_count = 3
    if(mode =="baseline"):
        for i in range(np):
            # system, core_num, action, index
            degree = [1, 0]
            for p in range(L2_prefetcher_count):
                m5.setL1RLDegree(testsys, i, degree[p], p)
                degrees.append(degree[p])
            for p in range(L1_prefetcher_count):
                m5.setL2RLDegree(testsys, i, degree[p], p)
                degrees.append(degree[p])
        for p in range(L1_prefetcher_count):
            L3_degree = [1,0,0]
            m5.setL3RLDegree(testsys, L3_degree[p], p)
            degrees.append(L3_degree[p])
    elif(mode =="multi"):
        for i in range(np):
            # system, core_num, action, index
            for p in range(L2_prefetcher_count):
                m5.setL1RLDegree(testsys, i, 1, p)
                degrees.append(1)
            for p in range(L1_prefetcher_count):
                m5.setL2RLDegree(testsys, i, 1, p)
                degrees.append(1)
        for p in range(L1_prefetcher_count):
            m5.setL3RLDegree(testsys, 1, p)
            degrees.append(1)
    elif(mode =="multid4"):
        for i in range(np):
            # system, core_num, action, index
            for p in range(L2_prefetcher_count):
                m5.setL1RLDegree(testsys, i, 4, p)
                degrees.append(4)
            for p in range(L1_prefetcher_count):
                m5.setL2RLDegree(testsys, i, 4, p)
                degrees.append(4)
        for p in range(L1_prefetcher_count):
            m5.setL3RLDegree(testsys, 4, p)
            degrees.append(4)
    elif(mode =="nopf"):
        for i in range(np):
            # system, core_num, action, index
            for p in range(L2_prefetcher_count):
                m5.setL1RLDegree(testsys, i, 0, p)
                degrees.append(0)
            for p in range(L1_prefetcher_count):
                m5.setL2RLDegree(testsys, i, 0, p)
                degrees.append(0)
        for p in range(L1_prefetcher_count):
            m5.setL3RLDegree(testsys, 0, p)
            degrees.append(0)
    elif(mode == "random"):
        for i in range(np):
            # system, core_num, action, index
            for p in range(L1_prefetcher_count): # There are 2 prefetchers per level
                l1_degree = randint(0, 5)
                degrees.append(l1_degree)
                m5.setL1RLDegree(testsys, i, l1_degree, p)
            for p in range(L2_prefetcher_count): # There are 2 prefetchers per level    
                l2_degree = randint(0, 5)
                degrees.append(l2_degree)
                m5.setL2RLDegree(testsys, i, l2_degree, p)
                
        for p in range(L3_prefetcher_count): # There are 3 prefetchers per level
            degree = randint(0, 5)
            degrees.append(degree)
            m5.setL3RLDegree(testsys, degree, p)
    elif(mode == "custom"):
        degree_s = (options.degrees).split(",")
        degrees_custom = []
        for d in degree_s:
            degrees_custom.append(int(d))
        
        if(options.alternate):
            global high_degree
            if(high_degree):
                for i in range(np):
                    # system, core_num, action, index
                    for p in range(L1_prefetcher_count): # There are 2 prefetchers per level
                        l1_degree = 5
                        degrees.append(l1_degree)
                        m5.setL1RLDegree(testsys, i, l1_degree, p)
                    for p in range(L2_prefetcher_count): # There are 2 prefetchers per level    
                        l2_degree = 5
                        degrees.append(l2_degree)
                        m5.setL2RLDegree(testsys, i, l2_degree, p)
                        
                for p in range(L3_prefetcher_count): # There are 3 prefetchers per level
                    degree = 5
                    degrees.append(degree)
                    m5.setL3RLDegree(testsys, degree, p)
            else:
                idx = 0
                
                for i in range(np):
                    # system, core_num, action, index
                    for p in range(L1_prefetcher_count): # There are 2 prefetchers per level
                      
                        l1_degree = degrees_custom[idx]
                        degrees.append(l1_degree)
                        m5.setL1RLDegree(testsys, i, l1_degree, p)
                        idx += 1
                        
                    for p in range(L2_prefetcher_count): # There are 2 prefetchers per level    
                        
                        l2_degree = degrees_custom[idx]
                        degrees.append(l2_degree)
                        m5.setL2RLDegree(testsys, i, l2_degree, p)
                        idx += 1
                        
                for p in range(L3_prefetcher_count): # There are 3 prefetchers per level
                    degree = degrees_custom[idx]
                    degrees.append(degree)
                    m5.setL3RLDegree(testsys, degree, p)
                    idx += 1
                    
        else:
            idx = 0
            for i in range(np):
                # system, core_num, action, index
                for p in range(L1_prefetcher_count): # There are 2 prefetchers per level
                    l1_degree = degrees_custom[idx]
                    degrees.append(l1_degree)
                    m5.setL1RLDegree(testsys, i, l1_degree, p)
                    idx += 1
                for p in range(L2_prefetcher_count): # There are 2 prefetchers per level    
                    l2_degree = degrees_custom[idx]
                    degrees.append(l2_degree)
                    m5.setL2RLDegree(testsys, i, l2_degree, p)
                    idx += 1
            for p in range(L3_prefetcher_count): # There are 3 prefetchers per level
                degree = degrees_custom[idx]
                degrees.append(degree)
                m5.setL3RLDegree(testsys, degree, p)
                idx += 1
            
    elif(mode == "RL"):
        degrees = take_action(state, options)
    else:
        print("No specific action to take")
    actions = (pd.DataFrame(degrees, index=components, columns =['degrees'])).T
    return actions
            
def restoreSimpointCheckpoint():
    exit_event = m5.simulate()
    exit_cause = exit_event.getCause()

    if exit_cause == "simpoint starting point found":
        print("Warmed up! Dumping and resetting stats!")
        # m5.stats.dump()
        m5.stats.reset()

        exit_event = m5.simulate()
        exit_cause = exit_event.getCause()

        if exit_cause == "simpoint starting point found":
            print("Done running SimPoint!")
            sys.exit(exit_event.getCode())

    print('Exiting @ tick %i because %s' % (m5.curTick(), exit_cause))
    sys.exit(exit_event.getCode())

def restoreSimpointCheckpoint_inference(options, testsys):
    print("******Running the model every ", options.sample_length)
    np = options.num_cpus
    name = options.app
    df = pd.DataFrame()
    model_name = options.model
    
    testsys.switch_cpus[0].setMaxInst(options.sample_length)
    exit_event = m5.simulate()
    exit_cause = exit_event.getCause()
    print("exit_cause", exit_cause)
    state, state_val = read_state(testsys, np, options.app, 0)
    m5.stats.reset()
    print("Warmup done")
    
    
    for sample in range(0, options.num_sample):
        print("***********Sample ", sample)
        m5.simulate(1000)
        testsys.switch_cpus[0].setMaxInst(options.sample_length)
        actions = apply_degree(testsys, options, state_val)
        # df = df.append(state_df)
        exit_event = m5.simulate()
        state, state_val = read_state(testsys, np, options.app, 0)
        exit_cause = exit_event.getCause()
        print("--------ITR DONE-------------",  exit_cause)
    
    # df.to_csv("/home/cc/goro/outputs/"+name+"_inference.csv")
    sys.exit(0)


def restoreSimpointCheckpoint_train(options, testsys):
    print("******Collecting samples started!")
    np = options.num_cpus
    name = options.app
    df = pd.DataFrame()
    
    testsys.switch_cpus[0].setMaxInst(options.sample_length)
    exit_event = m5.simulate()
    exit_cause = exit_event.getCause()
    print("exit_cause", exit_cause)
    state, _ = read_state(testsys, np, options.app, 0)
    print("state")
    print(state)
    print("Warmup done")
    
    global high_degree
    for sample in range(0, options.num_sample):
        print("***********Sample ", sample)
        m5.simulate(1000)
        testsys.switch_cpus[0].setMaxInst(options.sample_length)
        actions  = apply_degree(testsys, options, NULL)
        high_degree = not high_degree
        exit_event = m5.simulate()
        exit_cause = exit_event.getCause()
        print("exit_cause", exit_cause)
        next_state, _ = read_state(testsys, np, options.app, sample)
        df1 = dataset_create(state, next_state, actions, name+"."+str(sample))
        df = df.append(df1)
        state = next_state
        print("--------ITR DONE-------------")
    
    print("--------Time to exit-------------")
    df.to_csv("/home/cc/goro/csv/"+name+".csv")
    sys.exit(0)

def dataset_create(state, next_state, actions, name):
    print("dataset_create*************")
    state = state.add_prefix('S_')
    next_state = next_state.add_prefix('NS_')
    state = state.rename(index={state.index[0]: name})
    next_state = next_state.rename(index={next_state.index[0]: name})
    actions = actions.rename(index={actions.index[0]: name})
    df = pd.concat([state, next_state, actions], axis=1)
    return df
    
    
def repeatSwitch(testsys, repeat_switch_cpu_list, maxtick, switch_freq):
    print("starting switch loop")
    while True:
        exit_event = m5.simulate(switch_freq)
        exit_cause = exit_event.getCause()

        if exit_cause != "simulate() limit reached":
            return exit_event

        m5.switchCpus(testsys, repeat_switch_cpu_list)

        tmp_cpu_list = []
        for old_cpu, new_cpu in repeat_switch_cpu_list:
            tmp_cpu_list.append((new_cpu, old_cpu))
        repeat_switch_cpu_list = tmp_cpu_list

        if (maxtick - m5.curTick()) <= switch_freq:
            exit_event = m5.simulate(maxtick - m5.curTick())
            return exit_event

def run(options, root, testsys, cpu_class):
    if options.checkpoint_dir:
        cptdir = options.checkpoint_dir
    elif m5.options.outdir:
        cptdir = m5.options.outdir
    else:
        cptdir = getcwd()

    if options.fast_forward and options.checkpoint_restore != None:
        fatal("Can't specify both --fast-forward and --checkpoint-restore")

    if options.standard_switch and not options.caches:
        fatal("Must specify --caches when using --standard-switch")

    if options.standard_switch and options.repeat_switch:
        fatal("Can't specify both --standard-switch and --repeat-switch")

    if options.repeat_switch and options.take_checkpoints:
        fatal("Can't specify both --repeat-switch and --take-checkpoints")

    # Setup global stat filtering.
    stat_root_simobjs = []
    for stat_root_str in options.stats_root:
        stat_root_simobjs.extend(root.get_simobj(stat_root_str))
    m5.stats.global_dump_roots = stat_root_simobjs

    np = options.num_cpus
    switch_cpus = None

    if options.prog_interval:
        for i in range(np):
            testsys.cpu[i].progress_interval = options.prog_interval

    if options.maxinsts:
        for i in range(np):
            testsys.cpu[i].max_insts_any_thread = options.maxinsts

    if cpu_class:
        switch_cpus = [cpu_class(switched_out=True, cpu_id=(i))
                       for i in range(np)]

        for i in range(np):
            if options.fast_forward:
                testsys.cpu[i].max_insts_any_thread = int(options.fast_forward)
            switch_cpus[i].system = testsys
            switch_cpus[i].workload = testsys.cpu[i].workload
            switch_cpus[i].clk_domain = testsys.cpu[i].clk_domain
            switch_cpus[i].progress_interval = \
                testsys.cpu[i].progress_interval
            switch_cpus[i].isa = testsys.cpu[i].isa
            # simulation period
            if options.maxinsts:
                switch_cpus[i].max_insts_any_thread = options.maxinsts
            # Add checker cpu if selected
            if options.checker:
                switch_cpus[i].addCheckerCpu()
            if options.bp_type:
                bpClass = ObjectList.bp_list.get(options.bp_type)
                switch_cpus[i].branchPred = bpClass()
            if options.indirect_bp_type:
                IndirectBPClass = ObjectList.indirect_bp_list.get(
                    options.indirect_bp_type)
                switch_cpus[i].branchPred.indirectBranchPred = \
                    IndirectBPClass()

        # If elastic tracing is enabled attach the elastic trace probe
        # to the switch CPUs
        if options.elastic_trace_en:
            CpuConfig.config_etrace(cpu_class, switch_cpus, options)

        testsys.switch_cpus = switch_cpus
        switch_cpu_list = [(testsys.cpu[i], switch_cpus[i]) for i in range(np)]

    if options.repeat_switch:
        switch_class = getCPUClass(options.cpu_type)[0]
        if switch_class.require_caches() and \
                not options.caches:
            print("%s: Must be used with caches" % str(switch_class))
            sys.exit(1)
        if not switch_class.support_take_over():
            print("%s: CPU switching not supported" % str(switch_class))
            sys.exit(1)

        repeat_switch_cpus = [switch_class(switched_out=True, \
                                               cpu_id=(i)) for i in range(np)]

        for i in range(np):
            repeat_switch_cpus[i].system = testsys
            repeat_switch_cpus[i].workload = testsys.cpu[i].workload
            repeat_switch_cpus[i].clk_domain = testsys.cpu[i].clk_domain
            repeat_switch_cpus[i].isa = testsys.cpu[i].isa

            if options.maxinsts:
                repeat_switch_cpus[i].max_insts_any_thread = options.maxinsts

            if options.checker:
                repeat_switch_cpus[i].addCheckerCpu()

        testsys.repeat_switch_cpus = repeat_switch_cpus

        if cpu_class:
            repeat_switch_cpu_list = [(switch_cpus[i], repeat_switch_cpus[i])
                                      for i in range(np)]
        else:
            repeat_switch_cpu_list = [(testsys.cpu[i], repeat_switch_cpus[i])
                                      for i in range(np)]

    if options.standard_switch:
        switch_cpus = [TimingSimpleCPU(switched_out=True, cpu_id=(i))
                       for i in range(np)]
        switch_cpus_1 = [DerivO3CPU(switched_out=True, cpu_id=(i))
                        for i in range(np)]

        for i in range(np):
            switch_cpus[i].system =  testsys
            switch_cpus_1[i].system =  testsys
            switch_cpus[i].workload = testsys.cpu[i].workload
            switch_cpus_1[i].workload = testsys.cpu[i].workload
            switch_cpus[i].clk_domain = testsys.cpu[i].clk_domain
            switch_cpus_1[i].clk_domain = testsys.cpu[i].clk_domain
            switch_cpus[i].isa = testsys.cpu[i].isa
            switch_cpus_1[i].isa = testsys.cpu[i].isa

            # if restoring, make atomic cpu simulate only a few instructions
            if options.checkpoint_restore != None:
                testsys.cpu[i].max_insts_any_thread = 1
            # Fast forward to specified location if we are not restoring
            elif options.fast_forward:
                testsys.cpu[i].max_insts_any_thread = int(options.fast_forward)
            # Fast forward to a simpoint (warning: time consuming)
            elif options.simpoint:
                if testsys.cpu[i].workload[0].simpoint == 0:
                    fatal('simpoint not found')
                testsys.cpu[i].max_insts_any_thread = \
                    testsys.cpu[i].workload[0].simpoint
            # No distance specified, just switch
            else:
                testsys.cpu[i].max_insts_any_thread = 1

            # warmup period
            if options.warmup_insts:
                switch_cpus[i].max_insts_any_thread =  options.warmup_insts

            # simulation period
            if options.maxinsts:
                switch_cpus_1[i].max_insts_any_thread = options.maxinsts

            # attach the checker cpu if selected
            if options.checker:
                switch_cpus[i].addCheckerCpu()
                switch_cpus_1[i].addCheckerCpu()

        testsys.switch_cpus = switch_cpus
        testsys.switch_cpus_1 = switch_cpus_1
        switch_cpu_list = [
            (testsys.cpu[i], switch_cpus[i]) for i in range(np)
        ]
        switch_cpu_list1 = [
            (switch_cpus[i], switch_cpus_1[i]) for i in range(np)
        ]

    # set the checkpoint in the cpu before m5.instantiate is called
    if options.take_checkpoints != None and \
           (options.simpoint or options.at_instruction):
        offset = int(options.take_checkpoints)
        # Set an instruction break point
        if options.simpoint:
            for i in range(np):
                if testsys.cpu[i].workload[0].simpoint == 0:
                    fatal('no simpoint for testsys.cpu[%d].workload[0]', i)
                checkpoint_inst = int(testsys.cpu[i].workload[0].simpoint) + offset
                testsys.cpu[i].max_insts_any_thread = checkpoint_inst
                # used for output below
                options.take_checkpoints = checkpoint_inst
        else:
            options.take_checkpoints = offset
            # Set all test cpus with the right number of instructions
            # for the upcoming simulation
            for i in range(np):
                testsys.cpu[i].max_insts_any_thread = offset

    if options.take_simpoint_checkpoints != None:
        simpoints, interval_length = parseSimpointAnalysisFile(options, testsys)

    checkpoint_dir = None
    if options.checkpoint_restore:
        cpt_starttick, checkpoint_dir = findCptDir(options, cptdir, testsys)
    root.apply_config(options.param)
    m5.instantiate(checkpoint_dir)

    # Initialization is complete.  If we're not in control of simulation
    # (that is, if we're a slave simulator acting as a component in another
    #  'master' simulator) then we're done here.  The other simulator will
    # call simulate() directly. --initialize-only is used to indicate this.
    if options.initialize_only:
        return

    # Handle the max tick settings now that tick frequency was resolved
    # during system instantiation
    # NOTE: the maxtick variable here is in absolute ticks, so it must
    # include any simulated ticks before a checkpoint
    explicit_maxticks = 0
    maxtick_from_abs = m5.MaxTick
    maxtick_from_rel = m5.MaxTick
    maxtick_from_maxtime = m5.MaxTick
    if options.abs_max_tick:
        maxtick_from_abs = options.abs_max_tick
        explicit_maxticks += 1
    if options.rel_max_tick:
        maxtick_from_rel = options.rel_max_tick
        if options.checkpoint_restore:
            # NOTE: this may need to be updated if checkpoints ever store
            # the ticks per simulated second
            maxtick_from_rel += cpt_starttick
            if options.at_instruction or options.simpoint:
                warn("Relative max tick specified with --at-instruction or" \
                     " --simpoint\n      These options don't specify the " \
                     "checkpoint start tick, so assuming\n      you mean " \
                     "absolute max tick")
        explicit_maxticks += 1
    if options.maxtime:
        maxtick_from_maxtime = m5.ticks.fromSeconds(options.maxtime)
        explicit_maxticks += 1
    if explicit_maxticks > 1:
        warn("Specified multiple of --abs-max-tick, --rel-max-tick, --maxtime."\
             " Using least")
    maxtick = min([maxtick_from_abs, maxtick_from_rel, maxtick_from_maxtime])

    if options.checkpoint_restore != None and maxtick < cpt_starttick:
        fatal("Bad maxtick (%d) specified: " \
              "Checkpoint starts starts from tick: %d", maxtick, cpt_starttick)

    if options.standard_switch or cpu_class:
        if options.standard_switch:
            print("Switch at instruction count:%s" %
                    str(testsys.cpu[0].max_insts_any_thread))
            exit_event = m5.simulate()
        elif cpu_class and options.fast_forward:
            print("Switch at instruction count:%s" %
                    str(testsys.cpu[0].max_insts_any_thread))
            exit_event = m5.simulate()
        else:
            print("Switch at curTick count:%s" % str(10000))
            exit_event = m5.simulate(10000)
        print("Switched CPUS @ tick %s" % (m5.curTick()))

        m5.switchCpus(testsys, switch_cpu_list)

        if options.standard_switch:
            print("Switch at instruction count:%d" %
                    (testsys.switch_cpus[0].max_insts_any_thread))

            #warmup instruction count may have already been set
            if options.warmup_insts:
                exit_event = m5.simulate()
            else:
                exit_event = m5.simulate(options.standard_switch)
            print("Switching CPUS @ tick %s" % (m5.curTick()))
            print("Simulation ends instruction count:%d" %
                    (testsys.switch_cpus_1[0].max_insts_any_thread))
            m5.switchCpus(testsys, switch_cpu_list1)

    # If we're taking and restoring checkpoints, use checkpoint_dir
    # option only for finding the checkpoints to restore from.  This
    # lets us test checkpointing by restoring from one set of
    # checkpoints, generating a second set, and then comparing them.
    if (options.take_checkpoints or options.take_simpoint_checkpoints) \
        and options.checkpoint_restore:

        if m5.options.outdir:
            cptdir = m5.options.outdir
        else:
            cptdir = getcwd()

    if options.take_checkpoints != None :
        # Checkpoints being taken via the command line at <when> and at
        # subsequent periods of <period>.  Checkpoint instructions
        # received from the benchmark running are ignored and skipped in
        # favor of command line checkpoint instructions.
        exit_event = scriptCheckpoints(options, maxtick, cptdir)

    # Take SimPoint checkpoints
    elif options.take_simpoint_checkpoints != None:
        takeSimpointCheckpoints(simpoints, interval_length, cptdir)

    # Restore from SimPoint checkpoints
    elif options.restore_simpoint_checkpoint:
        if(options.train):
            print("--------train----")
            restoreSimpointCheckpoint_train(options, testsys)
        elif(options.inference):
            print("--------inference----")
            restoreSimpointCheckpoint_inference(options, testsys)
        
        else:
            print("--------Inference----")
            restoreSimpointCheckpoint()

    else:
        if options.fast_forward:
            m5.stats.reset()
        print("**** REAL SIMULATION ****")
        
        # If checkpoints are being taken, then the checkpoint instruction
        # will occur in the benchmark code it self.
        if options.repeat_switch and maxtick > options.repeat_switch:
            print("--------------1")
            exit_event = repeatSwitch(testsys, repeat_switch_cpu_list,
                                      maxtick, options.repeat_switch)
        else:
            print("----------Here")
            m5.simulate(10000000)
            state, state_val = read_state(testsys, np, options.app, 0)
            take_action(state_val, options)
            m5.simulate(10000000)
            state, state_val = read_state(testsys, np, options.app, 0)
            take_action(state_val, options)
            
            exit_event = benchCheckpoints(options, maxtick, cptdir)

    print('Exiting @ tick %i because %s' %
          (m5.curTick(), exit_event.getCause()))
    if options.checkpoint_at_end:
        m5.checkpoint(joinpath(cptdir, "cpt.%d"))

    if exit_event.getCode() != 0:
        print("Simulated exit code not 0! Exit code is", exit_event.getCode())
