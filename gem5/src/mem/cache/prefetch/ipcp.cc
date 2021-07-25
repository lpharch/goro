/*
 *
 * Ported from Champsim: Majid Jalili
 */

/**
 * @file
 * Describes a tagged prefetcher based on template policies.
 */

#include "mem/cache/prefetch/ipcp.hh"

#include "params/IPCPPrefetcher.hh"

#include "mem/cache/base.hh"
#include "cpu/base.hh"
#include "cpu/o3/cpu.hh"
#include "cpu/o3/impl.hh"

using namespace std;

namespace Prefetcher {
    
IPCPPrefetcher::IPCPPrefetcher(const IPCPPrefetcherParams &p)
    : Queued(p), degree(p.degree)
{
    totalPerf=0;
    l1d_prefetcher_initialize();
    // registerDumpCallback(
        // new MakeCallback<IPCPPrefetcher, &IPCPPrefetcher::onExit>(
            // this));

}

void
IPCPPrefetcher::onExit()
{
    l1d_prefetcher_final_stats();
}


uint16_t 
IPCPPrefetcher::update_sig_l1(uint16_t old_sig, int delta) {
    uint16_t new_sig = 0;
    int sig_delta = 0;

    // 7-bit sign magnitude form, since we need to track deltas from +63 to -63
    sig_delta = (delta < 0) ? (((-1) * delta) + (1 << 6)) : delta;
    new_sig = ((old_sig << 1) ^ sig_delta) & ((1 << NUM_SIG_BITS) - 1);

    return new_sig;
}


void
IPCPPrefetcher::l1d_prefetcher_initialize()
{
    int cpu = 0;
    for (int i = 0; i < NUM_RST_ENTRIES; i++)
        rstable[cpu][i].lru = i;
    for (int i = 0; i < NUM_CPUS; i++)
    {
        prefetch_degree[cpu][0] = 0;
        prefetch_degree[cpu][1] = 6;
        prefetch_degree[cpu][2] = 3;
        prefetch_degree[cpu][3] = 3;
        prefetch_degree[cpu][4] = 1;

    }
}

uint32_t 
IPCPPrefetcher::encode_metadata(int stride, uint16_t type, int spec_nl) {
    uint32_t metadata = 0;

    // first encode stride in the last 8 bits of the metadata
    if (stride > 0)
        metadata = stride;
    else
        metadata = ((-1 * stride) | 0b1000000);

    // encode the type of IP in the next 4 bits 			 
    metadata = metadata | (type << 8);

    // encode the speculative NL bit in the next 1 bit
    metadata = metadata | (spec_nl << 12);

    return metadata;
}

int 
IPCPPrefetcher::update_conf(int stride, int pred_stride, int conf) {
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


uint64_t 
IPCPPrefetcher::hash_bloom(uint64_t addr) {
    uint64_t first_half, sec_half;
    first_half = addr & 0xFFF;
    sec_half = (addr >> 12) & 0xFFF;
    if ((first_half ^ sec_half) >= 4096)
        assert(0);
    return ((first_half ^ sec_half) & 0xFFF);
}

uint64_t 
IPCPPrefetcher::hash_page(uint64_t addr) {
    uint64_t hash = 0;
    while (addr != 0) {
        hash = hash ^ addr;
        addr = addr >> 6;
    }

    return hash & ((1 << NUM_PAGE_TAG_BITS) - 1);
}


void
IPCPPrefetcher::calculatePrefetch(const PrefetchInfo &pfi,
        std::vector<AddrPriority> &addresses)
{
    
    int cache_hit = !pfi.isCacheMiss();

    FullO3CPU<O3CPUImpl> *o3cpu = dynamic_cast<FullO3CPU<O3CPUImpl> * >(cache->system->threads[0]->getCpuPtr());
    uint64_t totInst = o3cpu->baseStats.numCycles.value();
    for(int i = 0 ; i< o3cpu->cpuStats.committedInsts.size(); i++){
        totInst+=o3cpu->cpuStats.committedInsts[i].value();
    }
        

    // BaseCache* bcache = dynamic_cast<BaseCache*>(this->cache);
    // int64_t inst = bcache->numInst();
    int64_t inst = totInst;
    vector<tuple<uint64_t, uint64_t>> res = l1d_prefetcher_operate(pfi.getAddr(), pfi.getPC(), cache_hit, 0, inst);
    
    for (int d = 0; d < res.size(); d++) {
        Addr newAddr = blockAddress(std::get<0>(res[d]));
        Addr meta = std::get<1>(res[d]);
        addresses.push_back(AddrPriority(newAddr, meta));
    }
}

void 
IPCPPrefetcher::stat_col_L1(uint64_t addr, uint8_t cache_hit, uint8_t cpu, uint64_t ip) {
    uint64_t index = hash_bloom(addr);
    int ip_index = ip & ((1 << NUM_IP_INDEX_BITS) - 1);
    uint16_t ip_tag = (ip >> NUM_IP_INDEX_BITS) & ((1 << NUM_IP_TAG_BITS) - 1);


    for (int i = 0; i < 5; i++) {
        if (cache_hit) {
            if (stats[cpu][i].bl_filled[index] == 1) {
                stats[cpu][i].useful++;
                stats[cpu][i].filled++;
                stats[cpu][i].bl_filled[index] = 0;
            }
        }
        else {
            if (ip_tag == trackers_l1[cpu][ip_index].ip_tag) {
                if (trackers_l1[cpu][ip_index].pref_type == i)
                    stats[cpu][i].misses++;
                if (stats[cpu][i].bl_filled[index] == 1) {
                    stats[cpu][i].polluted_misses++;
                    stats[cpu][i].filled++;
                    stats[cpu][i].bl_filled[index] = 0;
                }
            }
        }

        if (num_misses[cpu] % 1024 == 0) {
            for (int j = 0; j < NUM_BLOOM_ENTRIES; j++) {
                stats[cpu][i].filled += stats[cpu][i].bl_filled[j];
                stats[cpu][i].bl_filled[j] = 0;
                stats[cpu][i].bl_request[j] = 0;
            }
        }
    }
}






void 
IPCPPrefetcher::l1d_prefetcher_final_stats()
{
    int cpu = 0;
    cout << endl;

    uint64_t total_request = 0, total_polluted = 0, total_useful = 0, total_late = 0;

    for (int i = 0; i < 5; i++) {
        total_request += pref_filled[cpu][i];
        total_polluted += stats[cpu][i].polluted_misses;
        total_useful += pref_useful[cpu][i];
        total_late += pref_late[cpu][i];
    }
    cout << "totalPerf " << totalPerf << endl;
    cout << "stream: " << endl;
    cout << "stream:times selected: " << meta_counter[cpu][0] << endl;
    cout << "stream:pref_filled: " << pref_filled[cpu][1] << endl;
    cout << "stream:pref_useful: " << pref_useful[cpu][1] << endl;
    cout << "stream:pref_late: " << pref_late[cpu][1] << endl;
    cout << "stream:misses: " << stats[cpu][1].misses << endl;
    cout << "stream:misses_by_poll: " << stats[cpu][1].polluted_misses << endl;
    cout << endl;

    cout << "CS: " << endl;
    cout << "CS:times selected: " << meta_counter[cpu][1] << endl;
    cout << "CS:pref_filled: " << pref_filled[cpu][2] << endl;
    cout << "CS:pref_useful: " << pref_useful[cpu][2] << endl;
    cout << "CS:pref_late: " << pref_late[cpu][2] << endl;
    cout << "CS:misses: " << stats[cpu][2].misses << endl;
    cout << "CS:misses_by_poll: " << stats[cpu][2].polluted_misses << endl;
    cout << endl;

    cout << "CPLX: " << endl;
    cout << "CPLX:times selected: " << meta_counter[cpu][2] << endl;
    cout << "CPLX:pref_filled: " << pref_filled[cpu][3] << endl;
    cout << "CPLX:pref_useful: " << pref_useful[cpu][3] << endl;
    cout << "CPLX:pref_late: " << pref_late[cpu][3] << endl;
    cout << "CPLX:misses: " << stats[cpu][3].misses << endl;
    cout << "CPLX:misses_by_poll: " << stats[cpu][3].polluted_misses << endl;
    cout << endl;

    cout << "NL_L1: " << endl;
    cout << "NL:times selected: " << meta_counter[cpu][3] << endl;
    cout << "NL:pref_filled: " << pref_filled[cpu][4] << endl;
    cout << "NL:pref_useful: " << pref_useful[cpu][4] << endl;
    cout << "NL:pref_late: " << pref_late[cpu][4] << endl;
    cout << "NL:misses: " << stats[cpu][4].misses << endl;
    cout << "NL:misses_by_poll: " << stats[cpu][4].polluted_misses << endl;
    cout << endl;


    cout << "total selections: " << total_count[cpu] << endl;
    //cout << "total_filled: " << pf_fill << endl;
   // cout << "total_useful: " << pf_useful << endl;
    cout << "total_late: " << total_late << endl;
    cout << "total_polluted: " << total_polluted << endl;
    cout << "total_misses_after_warmup: " << num_misses[cpu] << endl;

    cout << "conflicts: " << num_conflicts << endl;
    cout << endl;

    cout << "test: " << test << endl;
}


vector<tuple<uint64_t, uint64_t>>
IPCPPrefetcher::l1d_prefetcher_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, uint8_t type, int num_retired)
{
    int cpu = 0;
    vector<tuple<uint64_t, uint64_t>> candids;
    uint64_t curr_page = hash_page(addr >> LOG2_PAGE_SIZE); 	//current page 
    uint64_t line_addr = addr >> LOG2_BLOCK_SIZE;		//cache line address
    uint64_t line_offset = (addr >> LOG2_BLOCK_SIZE) & 0x3F; 	//cache line offset
    uint16_t signature = 0, last_signature = 0;
    int spec_nl_threshold = 0;
    int num_prefs = 0;
    uint32_t metadata = 0;
    uint16_t ip_tag = (ip >> NUM_IP_INDEX_BITS) & ((1 << NUM_IP_TAG_BITS) - 1);
    uint64_t bl_index = 0;

    if (NUM_CPUS == 1) {
        spec_nl_threshold = 50;
    }
    else {                                    //tightening the mpki constraints for multi-core
        spec_nl_threshold = 40;
    }


    if (cache_hit == 0 )
        num_misses[cpu] += 1;

    num_access[cpu] += 1;
    stat_col_L1(addr, cache_hit, cpu, ip);
    // update spec nl bit when num misses crosses certain threshold
    if (num_misses[cpu] % 256 == 0 && cache_hit == 0) {
        mpki[cpu] = ((num_misses[cpu] * 1000.0) / (num_retired )); //Check here Majid
        //cout << "mpki[cpu] " << mpki[cpu] << endl;
        if (mpki[cpu] > spec_nl_threshold)
            spec_nl[cpu] = 0;
        else
            spec_nl[cpu] = 1;
    }

    //Updating prefetch degree based on accuracy
    for (int i = 0; i < 5; i++)
    {
        if (pref_filled[cpu][i] % 256 == 0)
        {

            acc_useful[cpu][i] = acc_useful[cpu][i] / 2.0 + (pref_useful[cpu][i] - acc_useful[cpu][i]) / 2.0;
            acc_filled[cpu][i] = acc_filled[cpu][i] / 2.0 + (pref_filled[cpu][i] - acc_filled[cpu][i]) / 2.0;

            if (acc_filled[cpu][i] != 0)
                acc[cpu][i] = 100.0 * acc_useful[cpu][i] / (acc_filled[cpu][i]);
            else
                acc[cpu][i] = 60;

            if (acc[cpu][i] > 75)
            {
                prefetch_degree[cpu][i]++;
                if (i == 1)
                {
                    //For GS class, degree is incremented/decremented by 2.
                    prefetch_degree[cpu][i]++;
                    if (prefetch_degree[cpu][i] > 6)
                        prefetch_degree[cpu][i] = 6;
                }
                else if (prefetch_degree[cpu][i] > 3)
                    prefetch_degree[cpu][i] = 3;
            }
            else if (acc[cpu][i] < 40)
            {
                prefetch_degree[cpu][i]--;
                if (i == 1)
                    prefetch_degree[cpu][i]--;
                if (prefetch_degree[cpu][i] < 1)
                    prefetch_degree[cpu][i] = 1;
            }

        }
    }
    
    // calculate the index bit
    int index = ip & ((1 << NUM_IP_INDEX_BITS) - 1);
    if (trackers_l1[cpu][index].ip_tag != ip_tag) {               // new/conflict IP
        if (trackers_l1[cpu][index].ip_valid == 0) {              // if valid bit is zero, update with latest IP info
            num_conflicts++;
            trackers_l1[cpu][index].ip_tag = ip_tag;
            trackers_l1[cpu][index].last_vpage = curr_page;
            trackers_l1[cpu][index].last_line_offset = line_offset;
            trackers_l1[cpu][index].last_stride = 0;
            trackers_l1[cpu][index].signature = 0;
            trackers_l1[cpu][index].conf = 0;
            trackers_l1[cpu][index].str_valid = 0;
            trackers_l1[cpu][index].str_dir = 0;
            trackers_l1[cpu][index].pref_type = 0;
            trackers_l1[cpu][index].ip_valid = 1;
        }
        else {                                                    // otherwise, reset valid bit and leave the previous IP as it is
            trackers_l1[cpu][index].ip_valid = 0;
        }
       
        return candids;
    }
    else {                                                     // if same IP encountered, set valid bit
        trackers_l1[cpu][index].ip_valid = 1;
    }

    int64_t stride = 0;
    if (line_offset > trackers_l1[cpu][index].last_line_offset)
        stride = line_offset - trackers_l1[cpu][index].last_line_offset;
    else {
        stride = trackers_l1[cpu][index].last_line_offset - line_offset;
        stride *= -1;
    }

    // don't do anything if same address is seen twice in a row
    if (stride == 0) {
        
        return candids;
    }

   // cout << "stride  " << stride << endl;

    int c = 0, flag = 0;

    //Checking if IP is already classified as a part of the GS class, so that for the new region we will set the tentative (spec_dense) bit.
    for (int i = 0; i < NUM_RST_ENTRIES; i++)
    {
        if (rstable[cpu][i].region_id == ((trackers_l1[cpu][index].last_vpage << 1) | (trackers_l1[cpu][index].last_line_offset >> 5)))
        {
            if (rstable[cpu][i].trained_dense == 1)
                flag = 1;
            break;
        }
    }

    for (c = 0; c < NUM_RST_ENTRIES; c++)
    {
        if (((curr_page << 1) | (line_offset >> 5)) == rstable[cpu][c].region_id)
        {
            if (rstable[cpu][c].line_access[line_offset & REGION_OFFSET_MASK] == 0)
            {
                rstable[cpu][c].line_access[line_offset & REGION_OFFSET_MASK] = 1;
            }

            if (rstable[cpu][c].pos_neg_count >= MAX_POS_NEG_COUNT || rstable[cpu][c].pos_neg_count <= 0)
            {
                rstable[cpu][c].pos_neg_count = MAX_POS_NEG_COUNT / 2;
            }

            if (stride > 0)
                rstable[cpu][c].pos_neg_count++;
            else
                rstable[cpu][c].pos_neg_count--;

            if (rstable[cpu][c].trained_dense == 0)
            {
                int count = 0;
                for (int i = 0; i < NUM_OF_LINES_IN_REGION; i++)
                    if (rstable[cpu][c].line_access[line_offset & REGION_OFFSET_MASK] == 1)
                        count++;

                if (count > 24)	//75% of the cache lines in the region are accessed. 
                {
                    rstable[cpu][c].trained_dense = 1;
                }
            }
            if (flag == 1)
                rstable[cpu][c].tentative_dense = 1;
            if (rstable[cpu][c].tentative_dense == 1 || rstable[cpu][c].trained_dense == 1)
            {
                if (rstable[cpu][c].pos_neg_count > (MAX_POS_NEG_COUNT / 2))
                    rstable[cpu][c].dir = 1;	//1 for positive direction
                else
                    rstable[cpu][c].dir = 0;	//0 for negative direction
                trackers_l1[cpu][index].str_valid = 1;

                trackers_l1[cpu][index].str_dir = rstable[cpu][c].dir;
            }
            else
                trackers_l1[cpu][index].str_valid = 0;

            break;
        }
    }

    //curr page has no entry in rstable. Then replace lru.
    if (c == NUM_RST_ENTRIES)
    {
        //check lru
        for (c = 0; c < NUM_RST_ENTRIES; c++)
        {
            if (rstable[cpu][c].lru == (NUM_RST_ENTRIES - 1))
                break;
        }
        for (int i = 0; i < NUM_RST_ENTRIES; i++) {
            if (rstable[cpu][i].lru < rstable[cpu][c].lru)
                rstable[cpu][i].lru++;
        }
        if (flag == 1)
            rstable[cpu][c].tentative_dense = 1;
        else
            rstable[cpu][c].tentative_dense = 0;

        rstable[cpu][c].region_id = (curr_page << 1) | (line_offset >> 5);
        rstable[cpu][c].trained_dense = 0;
        rstable[cpu][c].pos_neg_count = MAX_POS_NEG_COUNT / 2;
        rstable[cpu][c].dir = 0;
        rstable[cpu][c].lru = 0;
        for (int i = 0; i < NUM_OF_LINES_IN_REGION; i++)
            rstable[cpu][c].line_access[i] = 0;
    }

    // page boundary learning
    if (curr_page != trackers_l1[cpu][index].last_vpage) {
        test++;
        if (stride < 0)
            stride += NUM_OF_LINES_IN_REGION;
        else
            stride -= NUM_OF_LINES_IN_REGION;
    }


    // update constant stride(CS) confidence
    trackers_l1[cpu][index].conf = update_conf(stride, trackers_l1[cpu][index].last_stride, trackers_l1[cpu][index].conf);

    // update CS only if confidence is zero
    if (trackers_l1[cpu][index].conf == 0)
        trackers_l1[cpu][index].last_stride = stride;
    

    last_signature = trackers_l1[cpu][index].signature;
    // update complex stride(CPLX) confidence
    CSPT_l1[cpu][last_signature].conf = update_conf(stride, CSPT_l1[cpu][last_signature].stride, CSPT_l1[cpu][last_signature].conf);

    // update CPLX only if confidence is zero
    if (CSPT_l1[cpu][last_signature].conf == 0)
        CSPT_l1[cpu][last_signature].stride = stride;

    // calculate and update new signature in IP table
    signature = update_sig_l1(last_signature, stride);
    trackers_l1[cpu][index].signature = signature;

   // cout << "stream IP " << trackers_l1[cpu][index].str_valid << endl;
    if (trackers_l1[cpu][index].str_valid == 1) {                          // stream IP
        // for stream, prefetch with twice the usual degree
        if (prefetch_degree[cpu][1] < 3)
            flag = 1;
        meta_counter[cpu][0]++;
        total_count[cpu]++;
        for (int i = 0; i < prefetch_degree[cpu][1]; i++) {
            uint64_t pf_address = 0;

            if (trackers_l1[cpu][index].str_dir == 1) {                   // +ve stream
                pf_address = (line_addr + i + 1) << LOG2_BLOCK_SIZE;
                metadata = encode_metadata(1, S_TYPE, spec_nl[cpu]);    // stride is 1
            }
            else {                                                       // -ve stream
                pf_address = (line_addr - i - 1) << LOG2_BLOCK_SIZE;
                metadata = encode_metadata(-1, S_TYPE, spec_nl[cpu]);   // stride is -1
            }

            if (acc[cpu][1] < 75)
                metadata = encode_metadata(0, S_TYPE, spec_nl[cpu]);
            // Check if prefetch address is in same 4 KB page
            if ((pf_address >> LOG2_PAGE_SIZE) != (addr >> LOG2_PAGE_SIZE)) {
                break;
            }

            trackers_l1[cpu][index].pref_type = S_TYPE;


            int found_in_filter = 0;
            for (int i = 0; i < recent_request_filter.size(); i++)
            {
                if (recent_request_filter[i] == ((pf_address >> 6) & RR_TAG_MASK))
                {
                    // Prefetch address is present in RR filter
                   // cout << "Found in filter" << endl;
                    found_in_filter = 1;
                }
            }
            //Issue prefetch request only if prefetch address is not present in RR filter
            if (found_in_filter == 0)
            {
                //prefetch_line(ip, addr, pf_address, FILL_L1, metadata);
                candids.push_back(std::make_tuple(pf_address, metadata)); //Majid
                totalPerf++;
               // cout << "pf_address " << pf_address << endl;
                //Add to RR filter
                recent_request_filter.push_back((pf_address >> 6) & RR_TAG_MASK);
                if (recent_request_filter.size() > NUM_OF_RR_ENTRIES)
                    recent_request_filter.erase(recent_request_filter.begin());
            }

            num_prefs++;
            //SIG_DP(cout << "1, ");
        }
    }else
        flag = 1;


    ///////////////////
   // cout << "CS IP  " << trackers_l1[cpu][index].conf <<" "<< trackers_l1[cpu][index].last_stride <<" "<< flag << endl;
    if (trackers_l1[cpu][index].conf > 1 && trackers_l1[cpu][index].last_stride != 0 && flag == 1) {            // CS IP  
        meta_counter[cpu][1]++;
        total_count[cpu]++;

        if (prefetch_degree[cpu][2] < 2)
            flag = 1;
        else
            flag = 0;

        for (int i = 0; i < prefetch_degree[cpu][2]; i++) {
            uint64_t pf_address = (line_addr + (trackers_l1[cpu][index].last_stride * (i + 1))) << LOG2_BLOCK_SIZE;

            // Check if prefetch address is in same 4 KB page
            if ((pf_address >> LOG2_PAGE_SIZE) != (addr >> LOG2_PAGE_SIZE)) {
                break;
            }

            trackers_l1[cpu][index].pref_type = CS_TYPE;
            bl_index = hash_bloom(pf_address);
            stats[cpu][CS_TYPE].bl_request[bl_index] = 1;
            if (acc[cpu][2] > 75)
                metadata = encode_metadata(trackers_l1[cpu][index].last_stride, CS_TYPE, spec_nl[cpu]);
            else
                metadata = encode_metadata(0, CS_TYPE, spec_nl[cpu]);
            // if(spec_nl[cpu] == 1)

            int found_in_filter = 0;
            for (int i = 0; i < recent_request_filter.size(); i++)
            {
                if (recent_request_filter[i] == ((pf_address >> 6) & RR_TAG_MASK))
                {
                    // Prefetch address is present in RR filter
                    found_in_filter = 1;
                }
            }
            //Issue prefetch request only if prefetch address is not present in RR filter
            if (found_in_filter == 0)
            {
                //prefetch_line(ip, addr, pf_address, FILL_L1, metadata);
                candids.push_back(std::make_tuple(pf_address, metadata)); //Majid
                totalPerf++;
                //Add to RR filter
                recent_request_filter.push_back((pf_address >> 6) & RR_TAG_MASK);
                if (recent_request_filter.size() > NUM_OF_RR_ENTRIES)
                    recent_request_filter.erase(recent_request_filter.begin());
            }

            num_prefs++;
            //SIG_DP(cout << trackers_l1[cpu][index].last_stride << ", ");
        }
    }
    else
        flag = 1;


    //////////////////////////////////
    //cout << "CPLX IP " << CSPT_l1[cpu][signature].conf <<" "<< CSPT_l1[cpu][signature].stride <<" "<<flag << endl;
    if (CSPT_l1[cpu][signature].conf >= 0 && CSPT_l1[cpu][signature].stride != 0 && flag == 1) {  // if conf>=0, continue looking for stride
        int pref_offset = 0, i = 0;                                                        // CPLX IP
        meta_counter[cpu][2]++;
        total_count[cpu]++;

        for (i = 0; i < prefetch_degree[cpu][3] + CPLX_DIST; i++) {
            pref_offset += CSPT_l1[cpu][signature].stride;
            uint64_t pf_address = ((line_addr + pref_offset) << LOG2_BLOCK_SIZE);

            // Check if prefetch address is in same 4 KB page
            if (((pf_address >> LOG2_PAGE_SIZE) != (addr >> LOG2_PAGE_SIZE)) ||
                (CSPT_l1[cpu][signature].conf == -1) ||
                (CSPT_l1[cpu][signature].stride == 0)) {
                // if new entry in CSPT or stride is zero, break
                break;
            }

            // we are not prefetching at L2 for CPLX type, so encode stride as 0
            trackers_l1[cpu][index].pref_type = CPLX_TYPE;
            metadata = encode_metadata(0, CPLX_TYPE, spec_nl[cpu]);
            if (CSPT_l1[cpu][signature].conf > 0 && i >= CPLX_DIST) {                                 // prefetch only when conf>0 for CPLX
                bl_index = hash_bloom(pf_address);
                stats[cpu][CPLX_TYPE].bl_request[bl_index] = 1;
                trackers_l1[cpu][index].pref_type = 3;

                int found_in_filter = 0;
                for (int i = 0; i < recent_request_filter.size(); i++)
                {
                    if (recent_request_filter[i] == ((pf_address >> 6) & RR_TAG_MASK))
                    {
                        // Prefetch address is present in RR filter
                        found_in_filter = 1;
                    }
                }
                //Issue prefetch request only if prefetch address is not present in RR filter
                if (found_in_filter == 0)
                {
                    //prefetch_line(ip, addr, pf_address, FILL_L1, metadata);
                    candids.push_back(std::make_tuple(pf_address, metadata)); //Majid
                    totalPerf++;
                    //Add to RR filter
                    recent_request_filter.push_back((pf_address >> 6) & RR_TAG_MASK);
                    if (recent_request_filter.size() > NUM_OF_RR_ENTRIES)
                        recent_request_filter.erase(recent_request_filter.begin());
                }

                num_prefs++;
                //SIG_DP(cout << pref_offset << ", ");
            }
            signature = update_sig_l1(signature, CSPT_l1[cpu][signature].stride);
        }
    }

    /////////////////
    // if no prefetches are issued till now, speculatively issue a next_line prefetch
    //cout << "NL " << num_prefs <<" "<<spec_nl[cpu] << endl;
    if (num_prefs == 0 && spec_nl[cpu] == 1) {
        if (flag_nl[cpu] == 0)
            flag_nl[cpu] = 1;
        else {
            uint64_t pf_address = ((addr >> LOG2_BLOCK_SIZE) + 1) << LOG2_BLOCK_SIZE;
            bl_index = hash_bloom(pf_address);
            stats[cpu][NL_TYPE].bl_request[bl_index] = 1;
            metadata = encode_metadata(1, NL_TYPE, spec_nl[cpu]);

            int found_in_filter = 0;
            for (int i = 0; i < recent_request_filter.size(); i++)
            {
                if (recent_request_filter[i] == ((pf_address >> 6) & RR_TAG_MASK))
                {
                    // Prefetch address is present in RR filter
                    found_in_filter = 1;
                }
            }
            //Issue prefetch request only if prefetch address is not present in RR filter
            if (found_in_filter == 0)
            {
                //prefetch_line(ip, addr, pf_address, FILL_L1, metadata);
                candids.push_back(std::make_tuple(pf_address, metadata)); //Majid
                totalPerf++;
                //Add to RR filter
                recent_request_filter.push_back((pf_address >> 6) & RR_TAG_MASK);
                if (recent_request_filter.size() > NUM_OF_RR_ENTRIES)
                    recent_request_filter.erase(recent_request_filter.begin());
            }

            trackers_l1[cpu][index].pref_type = NL_TYPE;
            meta_counter[cpu][3]++;
            total_count[cpu]++;
            //SIG_DP(cout << "1, ");

            if (acc[cpu][4] < 40)
                flag_nl[cpu] = 0;
        }                                       // NL IP
    }


    //SIG_DP(cout << endl);

    // update the IP table entries
    trackers_l1[cpu][index].last_line_offset = line_offset;
    trackers_l1[cpu][index].last_vpage = curr_page;

   // cout << "End line " << candids.size()<< endl;
    return candids;
}
}

