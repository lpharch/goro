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

#ifndef __MEM_CACHE_PREFETCH_IPCP_HH__
#define __MEM_CACHE_PREFETCH_IPCP_HH__

#include "mem/cache/prefetch/queued.hh"
#include "mem/packet.hh"



struct IPCPPrefetcherParams;
namespace Prefetcher {


#define DO_PREF 1
#define NUM_BLOOM_ENTRIES 4096				    // For book-keeping purposes
#define NUM_IP_TABLE_L1_ENTRIES 64	
#define NUM_CSPT_ENTRIES 128			   	    // = 2^NUM_SIG_BITS
#define NUM_SIG_BITS 7					    // num of bits in signature
#define NUM_IP_INDEX_BITS 6 	
#define NUM_IP_TAG_BITS 9 	
#define NUM_PAGE_TAG_BITS 2
#define S_TYPE 1                                            // stream
#define CS_TYPE 2                                           // constant stride
#define CPLX_TYPE 3                                         // complex stride
#define NL_TYPE 4                                           // next line
#define CPLX_DIST 0                                          
#define NUM_OF_RR_ENTRIES 32				    // recent request filter entries
#define RR_TAG_MASK 0xFFF				    // 12 bits of prefetch line address are stored in recent request filter
#define NUM_RST_ENTRIES 8                                   // region stream table entries      
#define MAX_POS_NEG_COUNT 64				    // 6-bit saturating counter
#define NUM_OF_LINES_IN_REGION 32			    // 32 cache lines in 2KB region
#define REGION_OFFSET_MASK 0x1F				    // 5-bit offset for 2KB region
#define NUM_CPUS 1
#define LOG2_PAGE_SIZE 12 //Check this Majid
#define LOG2_BLOCK_SIZE 6 //Check this Majid

class IPCPPrefetcher : public Queued
{
    class IP_TABLE_L1 {
    public:
        uint64_t ip_tag;
        uint64_t last_vpage;                                    // last page seen by IP 
        uint64_t last_line_offset;                              // last cl offset in the 4KB page 
        int64_t last_stride;                                    // last stride observed 
        uint16_t ip_valid;					    // valid bit
        int conf;                                               // CS confidence 
        uint16_t signature;                                     // CPLX signature 
        uint16_t str_dir;                                       // stream direction 
        uint16_t str_valid;                                     // stream valid 
        uint16_t pref_type;                                     // pref type or class for book-keeping purposes.

        IP_TABLE_L1() {
            ip_tag = 0;
            last_vpage = 0;
            last_line_offset = 0;
            last_stride = 0;
            ip_valid = 0;
            signature = 0;
            conf = 0;
            str_dir = 0;
            str_valid = 0;
            pref_type = 0;
        };
    };

    class CONST_STRIDE_PRED_TABLE {
    public:
        int stride;
        int conf;

        CONST_STRIDE_PRED_TABLE() {
            stride = 0;
            conf = 0;
        };
    };

    class REGION_STREAM_TABLE {
    public:
        uint64_t region_id;
        uint64_t tentative_dense;			// tentative dense bit
        uint64_t trained_dense;				// trained dense bit
        uint64_t pos_neg_count;				// positive/negative stream counter
        uint64_t dir;					// direction of stream - 1 for +ve and 0 for -ve
        uint64_t lru;					// lru for replacement
        uint8_t line_access[NUM_OF_LINES_IN_REGION];	// bit vector to store which lines in the 2KB region have been accessed
        REGION_STREAM_TABLE() {
            region_id = 0;
            tentative_dense = 0;
            trained_dense = 0;
            pos_neg_count = MAX_POS_NEG_COUNT / 2;
            dir = 0;
            lru = 0;
            for (int i = 0; i < NUM_OF_LINES_IN_REGION; i++)
                line_access[i] = 0;
        };
    };
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
  protected:
      const int degree;

  public:
      IPCPPrefetcher(const IPCPPrefetcherParams &p);
      uint16_t update_sig_l1(uint16_t old_sig, int delta);
      uint32_t encode_metadata(int stride, uint16_t type, int spec_nl);
      int update_conf(int stride, int pred_stride, int conf);
      uint64_t hash_bloom(uint64_t addr);
      uint64_t hash_page(uint64_t addr);
      void stat_col_L1(uint64_t addr, uint8_t cache_hit, uint8_t cpu, uint64_t ip);
      void l1d_prefetcher_initialize();
      vector<tuple<uint64_t, uint64_t>> l1d_prefetcher_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, int inst);
      void l1d_prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in);
      void l1d_prefetcher_final_stats();
      void onExit();

      REGION_STREAM_TABLE rstable[NUM_CPUS][NUM_RST_ENTRIES];
      int acc_filled[NUM_CPUS][5];
      int acc_useful[NUM_CPUS][5];

      int acc[NUM_CPUS][5];
      int prefetch_degree[NUM_CPUS][5];
      int num_conflicts = 0;
      int test;
      int totalPerf;

      uint64_t eval_buffer[NUM_CPUS][1024] = {};
      STAT_COLLECT stats[NUM_CPUS][5];     // for GS, CS, CPLX, NL and no class
      IP_TABLE_L1 trackers_l1[NUM_CPUS][NUM_IP_TABLE_L1_ENTRIES];
      CONST_STRIDE_PRED_TABLE CSPT_l1[NUM_CPUS][NUM_CSPT_ENTRIES];

      uint64_t prev_cpu_cycle[NUM_CPUS];
      uint64_t num_misses[NUM_CPUS];
      float mpki[NUM_CPUS] = {  };
      int spec_nl[NUM_CPUS] = {  }, flag_nl[NUM_CPUS] = {  };
      uint64_t num_access[NUM_CPUS];

      int meta_counter[NUM_CPUS][4] = {  };                                                  // for book-keeping
      int total_count[NUM_CPUS] = {  };                                                  // for book-keeping

      vector<uint64_t> recent_request_filter;		// to filter redundant prefetch requests 


      int pref_filled[NUM_CPUS][6];
      int pref_late[NUM_CPUS][6];
      int pref_useful[NUM_CPUS][6];




    ~IPCPPrefetcher() {}

    void calculatePrefetch(const PrefetchInfo &pfi,
                           std::vector<AddrPriority> &addresses) override;
};
}
#endif // __MEM_CACHE_PREFETCH_IPCP_HH__