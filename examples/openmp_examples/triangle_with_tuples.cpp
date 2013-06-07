#include "A.h"
#include <omp.h>
#include "DictOut.hpp"
#include <iostream>

#define BILLION 1000000000
#if defined(__MTA__)
#include <sys/mta_task.h>
#include <machine/runtime.h>
#elif defined(__MACH__)
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

/// "Universal" wallclock time (works at least for Mac, MTA, and most Linux)
inline double walltime(void) {
#if defined(__MTA__)
	return((double)mta_get_clock(0) / mta_clock_freq());
#elif defined(__MACH__)
	static mach_timebase_info_data_t info;
	mach_timebase_info(&info);
	uint64_t now = mach_absolute_time();
	now *= info.numer;
	now /= info.denom;
	return 1.0e-9 * (double)now;
#else
	struct timespec tp;
#if defined(CLOCK_PROCESS_CPUTIME_ID)
#define CLKID CLOCK_PROCESS_CPUTIME_ID
#elif  defined(CLOCK_REALTIME_ID)
#define CLKID CLOCK_REALTIME_ID
#endif
	clock_gettime(CLOCK_MONOTONIC, &tp);
	return (double)tp.tv_sec + (double)tp.tv_nsec / BILLION;
#endif
}


struct tuple {
    int to;
    int from;

    bool operator<(const tuple& rhs) const {
        return to < rhs.to;
    }

    bool operator==(const tuple& rhs) const {
        return to == rhs.to && from == rhs.from;
    }
};

double scan_runtime,
       hash_runtime,
       triangles_runtime;

void query (const char* fname,int num_threads) {

    int result = 0;
    
    
    double start, end; 
    start = walltime();
    
    //scan edges
    vector<tuple> edges = vector<tuple>();
    ifstream f0(fname);
    while (!f0.eof()) {
    	int j;
    	f0 >> j;
        int k;
        f0 >> k;
        tuple t; t.to = j; t.from = k;
        edges.push_back(t);
    	//tmp_vector0.push_back(j);
    	//count0++;
    	//if (count0 == 2) {
    	//	count0 = 0;
    	//	edges.push_back(tmp_vector0);
    	//	tmp_vector0 = vector<int>();
    	//}
    }
    f0.close();
    
    end = walltime();
    scan_runtime = end - start;
    
    
    cout << "done reading file.\n";
    
    start = walltime();

    //hash edges
    map<int, vector<tuple > > edges0_hash;
    for (int i = 0; i < edges.size(); i++) {
    	if (edges0_hash.find(edges[i].to) == edges0_hash.end()) {
    		edges0_hash[edges[i].to] = vector<tuple> ();
    	}
    	edges0_hash[edges[i].to].push_back(edges[i]);
    }

    end = walltime();
    hash_runtime = end - start;
    
    
    cout << "done creating hash.\n";

    omp_set_num_threads(num_threads);
    
    start = walltime();
    
    //loop over edges
    #pragma omp parallel for reduction(+:result) schedule(static)
    for (int index0 = 0; index0 < edges.size(); ++index0) {
        if (edges[index0].to > edges[index0].from) { continue; }
        //if there is no match, continue
        if (edges0_hash.find(edges[index0].from) == edges0_hash.end()) {
            continue;
        }
        vector<tuple> table1 = edges0_hash[edges[index0].from];
    
    
    
        //loop over table1
        #pragma omp parallel for reduction(+:result) schedule(static)
        for (int index1 = 0; index1 < table1.size(); ++index1) {
            if (table1[index1].to > table1[index1].from) { continue;}
            //if there is no match, continue
            if (edges0_hash.find(table1[index1].from) == edges0_hash.end()) {
                continue;
            }
            vector<tuple> table2 = edges0_hash[table1[index1].from];
        
        
        
            //loop over final join results
            #pragma omp parallel for reduction(+:result) schedule(static)
            for (int index2 = 0; index2 < table2.size(); ++index2) {
                if (table2[index2].from==edges[index0].to) {
                        ++result;
                        
                        
                }
            }
        }
    }
    
    end = walltime();
    triangles_runtime = end - start;

    cout << "Found " << result << " tuples.\n";

    char scheduling[1024] = "static";
    int64_t chunk = -1;
    
    DictOut out;
    DICT_ADD(out, hash_runtime);
    DICT_ADD(out, triangles_runtime);
    DICT_ADD(out, scan_runtime);
    DICT_ADD(out, (int64_t)num_threads);
    DICT_ADD(out, fname);
    DICT_ADD(out, scheduling);
    DICT_ADD(out, chunk);
    std::cout << out.toString() << std::endl; 
}

int main(int argc, const char* argv[]) { 
    if (argc < 3) {
        cout < "Usage: <exec> file-name num_threads\n";
        exit(1);
    }
    const char* fname = argv[1];
    int num_threads = atoi(argv[2]);

    query(fname,num_threads);
}
