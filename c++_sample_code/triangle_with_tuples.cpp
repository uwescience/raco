#include "A.h"
#include <omp.h>
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

void query (const char* fname,int num_threads) {

    int result = 0;
    
    
    
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
    
    
    cout << "done reading file.\n";
    
    //hash edges
    map<int, vector<tuple > > edges0_hash;
    for (int i = 0; i < edges.size(); i++) {
    	if (edges0_hash.find(edges[i].to) == edges0_hash.end()) {
    		edges0_hash[edges[i].to] = vector<tuple> ();
    	}
    	edges0_hash[edges[i].to].push_back(edges[i]);
    }
    
    cout << "done creating hash.\n";

    omp_set_num_threads(num_threads);
    
    
    //loop over edges
    #pragma omp parallel for reduction(+:result) schedule(dynamic,1)
    for (int index0 = 0; index0 < edges.size(); ++index0) {
        if (edges[index0].to > edges[index0].from) { continue; }
        //if there is no match, continue
        if (edges0_hash.find(edges[index0].from) == edges0_hash.end()) {
            continue;
        }
        vector<tuple> table1 = edges0_hash[edges[index0].from];
    
    
    
        //loop over table1
        #pragma omp parallel for reduction(+:result) schedule(dynamic,1)
        for (int index1 = 0; index1 < table1.size(); ++index1) {
            if (table1[index1].to > table1[index1].from) { continue;}
            //if there is no match, continue
            if (edges0_hash.find(table1[index1].from) == edges0_hash.end()) {
                continue;
            }
            vector<tuple> table2 = edges0_hash[table1[index1].from];
        
        
        
            //loop over final join results
            #pragma omp parallel for reduction(+:result) schedule(dynamic,1)
            for (int index2 = 0; index2 < table2.size(); ++index2) {
                if (table2[index2].from==edges[index0].to) {
                        ++result;
                        
                        
                }
            }
        }
    }

    cout << "Found " << result << " tuples.\n";
    
    
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
