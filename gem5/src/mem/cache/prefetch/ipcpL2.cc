/*
 * Copyright (c) 2005 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Ron Dreslinski
 */

/**
 * @file
 * Describes a tagged prefetcher based on template policies.
 */

#include "mem/cache/prefetch/ipcpL2.hh"

#include "params/IPCPL2Prefetcher.hh"
#include "mem/cache/base.hh"

using namespace std;

namespace Prefetcher {
IPCPL2Prefetcher::IPCPL2Prefetcher(const IPCPL2PrefetcherParams &p)
    : Queued(p), degree(p.degree)
{

}

void
IPCPL2Prefetcher::calculatePrefetch(const PrefetchInfo &pfi,
        std::vector<AddrPriority> &addresses)
{
    vector<tuple<uint64_t, uint64_t>> res;
    int cache_hit = !pfi.isCacheMiss();
    if (pfi.hasPC()) {
        res = l2c_prefetcher_operate(pfi.getAddr(), pfi.getPC(), cache_hit, 0, pfi.metadataICMP);
    }
    

    for (int d = 0; d < res.size(); d++) {
        Addr newAddr = blockAddress(std::get<0>(res[d]));
        Addr meta = std::get<1>(res[d]);
        addresses.push_back(AddrPriority(newAddr, meta));
    }
}



uint64_t 
IPCPL2Prefetcher::hash_bloom_l2(uint64_t addr) {
    uint64_t first_half, sec_half;
    first_half = addr & 0xFFF;
    sec_half = (addr >> 12) & 0xFFF;
    if ((first_half ^ sec_half) >= 4096)
        assert(0);
    return ((first_half ^ sec_half) & 0xFFF);
}

/*decode_stride: This function decodes 7 bit stride from the metadata from IPCP at L1. 6 bits for magnitude and 1 bit for sign. */

int 
IPCPL2Prefetcher::decode_stride(uint32_t metadata) {
    int stride = 0;
    if (metadata & 0b1000000)
        stride = -1 * (metadata & 0b111111);
    else
        stride = metadata & 0b111111;

    return stride;
}

/* update_conf_l2: If the actual stride and predicted stride are equal, then the confidence counter is incremented. */

int 
IPCPL2Prefetcher::update_conf_l1(int stride, int pred_stride, int conf) {
    if (stride == pred_stride) {             // use 2-bit saturating counter for confidence
        conf++;
        if (conf > 3)
            conf = 3;
    }
    else {
        conf--;
        if (conf < 0)
            conf = 0;
    }

    return conf;
}

uint32_t 
IPCPL2Prefetcher::encode_metadata_l2(int stride, uint16_t type, int spec_nl_l2) {

    uint32_t metadata = 0;

    // first encode stride in the last 8 bits of the metadata
    if (stride > 0)
        metadata = stride;
    else
        metadata = ((-1 * stride) | 0b1000000);

    // encode the type of IP in the next 4 bits
    metadata = metadata | (type << 8);

    // encode the speculative NL bit in the next 1 bit
    metadata = metadata | (spec_nl_l2 << 12);

    return metadata;

}


void 
IPCPL2Prefetcher::stat_col_L2(uint64_t addr, uint8_t cache_hit, uint8_t cpu, uint64_t ip) {
    uint64_t index = hash_bloom_l2(addr);
    int ip_index = ip & ((1 << NUM_IP_INDEX_BITS_L2) - 1);
    uint16_t ip_tag = (ip >> NUM_IP_INDEX_BITS_L2) & ((1 << NUM_IP_TAG_BITS_L2) - 1);

    for (int i = 0; i < 5; i++) {
        if (cache_hit) {
            if (stats_l2[cpu][i].bl_filled[index] == 1) {
                stats_l2[cpu][i].useful++;
                stats_l2[cpu][i].filled++;
                stats_l2[cpu][i].bl_filled[index] = 0;
            }
        }
        else {
            if (ip_tag == trackers[cpu][ip_index].ip_tag) {
                if (trackers[cpu][ip_index].pref_type == i)
                    stats_l2[cpu][i].misses++;
                if (stats_l2[cpu][i].bl_filled[index] == 1) {
                    stats_l2[cpu][i].polluted_misses++;
                    stats_l2[cpu][i].filled++;
                    stats_l2[cpu][i].bl_filled[index] = 0;
                }
            }
        }

        if (num_misses_l2[cpu] % 1024 == 0) {
            for (int j = 0; j < NUM_BLOOM_ENTRIES; j++) {
                stats_l2[cpu][i].filled += stats_l2[cpu][i].bl_filled[j];
                stats_l2[cpu][i].bl_filled[j] = 0;
                stats_l2[cpu][i].bl_request[j] = 0;
            }
        }
    }
}


uint32_t 
IPCPL2Prefetcher::l2c_prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
    int cpu = 0;
    if (prefetch) {
        uint32_t pref_type = metadata_in & 0xF00;
        pref_type = pref_type >> 8;

        uint64_t index = hash_bloom_l2(addr);
        if (stats_l2[cpu][pref_type].bl_request[index] == 1) {
            stats_l2[cpu][pref_type].bl_filled[index] = 1;
            stats_l2[cpu][pref_type].bl_request[index] = 0;
        }
    }
    return 0;
}


vector<tuple<uint64_t, uint64_t>>
IPCPL2Prefetcher::l2c_prefetcher_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in)
{
    int cpu = 0;
    vector<tuple<uint64_t, uint64_t>> candids;
    //uint64_t page = addr >> LOG2_PAGE_SIZE;
    //uint64_t curr_tag = (page ^ (page >> 6) ^ (page >> 12)) & ((1 << NUM_IP_TAG_BITS_L2) - 1);
    //uint64_t line_offset = (addr >> LOG2_BLOCK_SIZE) & 0x3F;
    uint64_t line_addr = addr >> LOG2_BLOCK_SIZE;
    int prefetch_degree = 0;
    int64_t stride = decode_stride(metadata_in);
    uint32_t pref_type = (metadata_in & 0xF00) >> 8;
    uint16_t ip_tag = (ip >> NUM_IP_INDEX_BITS_L2) & ((1 << NUM_IP_TAG_BITS_L2) - 1);
    int num_prefs = 0;
    uint64_t bl_index = 0;
    if (NUM_CPUS == 1) {
        prefetch_degree = 3;
    }
    else {                                    // tightening the degree for multi-core
        prefetch_degree = 2;
    }

    stat_col_L2(addr, cache_hit, cpu, ip);
    if (cache_hit == 0 && type != PREFETCH)
        num_misses_l2[cpu]++;

    // calculate the index bit
    int index = ip & ((1 << NUM_IP_INDEX_BITS_L2) - 1);
    if (trackers[cpu][index].ip_tag != ip_tag) {              // new/conflict IP
        if (trackers[cpu][index].ip_valid == 0) {             // if valid bit is zero, update with latest IP info
            trackers[cpu][index].ip_tag = ip_tag;
            trackers[cpu][index].pref_type = pref_type;
            trackers[cpu][index].stride = stride;
        }
        else {
            trackers[cpu][index].ip_valid = 0;                  // otherwise, reset valid bit and leave the previous IP as it is
        }

        // issue a next line prefetch upon encountering new IP
        uint64_t pf_address = ((addr >> LOG2_BLOCK_SIZE) + 1) << LOG2_BLOCK_SIZE;

        //prefetch_line(ip, addr, pf_address, FILL_L2, metadata);
        candids.push_back(std::make_tuple(pf_address, metadata_in)); //Majid

        SIG_DP(cout << "1, ");
        //
        //candids.push_back(metadata_in);//Majid
        return candids;
    }
    else {                                                  // if same IP encountered, set valid bit
        trackers[cpu][index].ip_valid = 1;
    }

    // update the IP table upon receiving metadata from prefetch
    if (type == PREFETCH) {
        trackers[cpu][index].pref_type = pref_type;
        trackers[cpu][index].stride = stride;
        spec_nl_l2[cpu] = metadata_in & 0x1000;
    }

    SIG_DP(
        cout << ip << ", " << cache_hit << ", " << line_addr << ", ";
    cout << ", " << stride << "; ";
    );


    if ((trackers[cpu][index].pref_type == 1 || trackers[cpu][index].pref_type == 2) && trackers[cpu][index].stride != 0) {      // S or CS class   
        uint32_t metadata = 0;
        if (trackers[cpu][index].pref_type == 1) {
            prefetch_degree = prefetch_degree * 2;
            metadata = encode_metadata_l2(1, S_TYPE, spec_nl_l2[cpu]);                                // for stream, prefetch with twice the usual degree
        }
        else {
            metadata = encode_metadata_l2(1, CS_TYPE, spec_nl_l2[cpu]);                                // for stream, prefetch with twice the usual degree
        }

        for (int i = 0; i < prefetch_degree; i++) {
            uint64_t pf_address = (line_addr + (trackers[cpu][index].stride * (i + 1))) << LOG2_BLOCK_SIZE;

            // Check if prefetch address is in same 4 KB page
            if ((pf_address >> LOG2_PAGE_SIZE) != (addr >> LOG2_PAGE_SIZE))
                break;
            num_prefs++;
            //prefetch_line(ip, addr, pf_address, FILL_L2, metadata);
            candids.push_back(std::make_tuple(pf_address, metadata)); //Majid
            SIG_DP(cout << trackers[cpu][index].stride << ", ");
        }
    }


    // if no prefetches are issued till now, speculatively issue a next_line prefetch
    if (num_prefs == 0 && spec_nl_l2[cpu] == 1) {                                        // NL IP
        uint64_t pf_address = ((addr >> LOG2_BLOCK_SIZE) + 1) << LOG2_BLOCK_SIZE;
        bl_index = hash_bloom_l2(pf_address);
        stats_l2[cpu][NL_TYPE].bl_request[bl_index] = 1;
        uint32_t metadata = encode_metadata_l2(1, NL_TYPE, spec_nl_l2[cpu]);
        trackers[cpu][index].pref_type = 3;
        //prefetch_line(ip, addr, pf_address, FILL_L2, metadata);
        candids.push_back(std::make_tuple(pf_address, metadata)); //Majid

        SIG_DP(cout << "1, ");
    }


    SIG_DP(cout << endl);
    //candids.push_back(metadata_in);//Majid
    return candids;
}
}

