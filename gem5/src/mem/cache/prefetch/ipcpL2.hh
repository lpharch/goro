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
 * Describes a tagged prefetcher.
 */

#ifndef __MEM_CACHE_PREFETCH_IPCPL2_HH__
#define __MEM_CACHE_PREFETCH_IPCPL2_HH__

#include "mem/cache/prefetch/queued.hh"
#include "mem/packet.hh"

struct IPCPL2PrefetcherParams;

namespace Prefetcher {




class IPCPL2Prefetcher : public Queued
{
#define DO_PREF 1

#define NUM_BLOOM_ENTRIES 4096
#define NUM_IP_TABLE_L2_ENTRIES 64
#define NUM_IP_INDEX_BITS_L2 6
#define NUM_IP_TAG_BITS_L2 9
#define S_TYPE 1                                            // stream
#define CS_TYPE 2                                           // constant stride
#define CPLX_TYPE 3                                         // complex stride
#define NL_TYPE 4                                           // next line
#define NUM_CPUS 1 //mCheck this Majid
#define LOG2_PAGE_SIZE 12 //Check this Majid
#define LOG2_BLOCK_SIZE 6 //Check this Majid

    //Majid really check this
#define PREFETCH 2



    // #define SIG_DEBUG_PRINT_L2				    //Uncomment to enable debug prints
#ifdef SIG_DEBUG_PRINT_L2
#define SIG_DP(x) x
#else
#define SIG_DP(x)
#endif

    class STAT_COLLECT {
    public:
        uint64_t useful;
        uint64_t filled;
        uint64_t misses;
        uint64_t polluted_misses;

        uint8_t bl_filled[NUM_BLOOM_ENTRIES];
        uint8_t bl_request[NUM_BLOOM_ENTRIES];

        STAT_COLLECT() {
            useful = 0;
            filled = 0;
            misses = 0;
            polluted_misses = 0;

            for (int i = 0; i < NUM_BLOOM_ENTRIES; i++) {
                bl_filled[i] = 0;
                bl_request[i] = 0;
            }
        };
    };

    class IP_TABLE {
    public:
        uint64_t ip_tag;						// ip tag
        uint16_t ip_valid;						// ip valid bit
        uint32_t pref_type;                                     	// prefetch class type
        int stride;							// stride or stream

        IP_TABLE() {
            ip_tag = 0;
            ip_valid = 0;
            pref_type = 0;
            stride = 0;
        };
    };

    

  protected:
      const int degree;

  public:
    IPCPL2Prefetcher(const IPCPL2PrefetcherParams &p);

    STAT_COLLECT stats_l2[NUM_CPUS][5];     // for GS, CS, CPLX, NL and no class
    uint64_t num_misses_l2[NUM_CPUS] = { 0 };
    //DELTA_PRED_TABLE CSPT_l2[NUM_CPUS][NUM_CSPT_L2_ENTRIES];
    uint32_t spec_nl_l2[NUM_CPUS] = { 0 };
    IP_TABLE trackers[NUM_CPUS][NUM_IP_TABLE_L2_ENTRIES];

    uint64_t hash_bloom_l2(uint64_t addr);
    int decode_stride(uint32_t metadata);
    int update_conf_l1(int stride, int pred_stride, int conf);
    uint32_t encode_metadata_l2(int stride, uint16_t type, int spec_nl_l2);
    void stat_col_L2(uint64_t addr, uint8_t cache_hit, uint8_t cpu, uint64_t ip);
    vector<tuple<uint64_t, uint64_t>> l2c_prefetcher_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, uint32_t metadata_in);
    uint32_t l2c_prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in);


    ~IPCPL2Prefetcher() {}

    void calculatePrefetch(const PrefetchInfo &pfi,
                           std::vector<AddrPriority> &addresses) override;
};
}
#endif // __MEM_CACHE_PREFETCH_IPCPL2_HH__