#include "A.h"

void query () {

    int result = 0;
    
    
    
    //scan edges
    vector<vector<int> > edges = vector<vector<int> >();
    ifstream f0("edges");
    int count0 = 0;
    vector<int> tmp_vector0 = vector<int>();
    while (!f0.eof()) {
    	int j;
    	f0 >> j;
    	tmp_vector0.push_back(j);
    	count0++;
    	if (count0 == 2) {
    		count0 = 0;
    		edges.push_back(tmp_vector0);
    		tmp_vector0 = vector<int>();
    	}
    }
    f0.close();
    
    
    
    
    //hash edges
    map<int, vector<vector<int> > > edges1_hash;
    for (int i = 0; i < edges.size(); i++) {
    	if (edges1_hash.find(edges[i][1]) == edges1_hash.end()) {
    		edges1_hash[edges[i][1]] = vector<vector<int> > ();
    	}
    	edges1_hash[edges[i][1]].push_back(edges[i]);
    }
    
    
    
    
    //loop over edges
    for (int index0 = 0; index0 < edges.size(); ++index0) {
        //if there is no match, continue
        if (edges1_hash.find(edges[index0][0]) == edges1_hash.end()) {
            continue;
        }
        vector<vector<int> > table1 = edges1_hash[edges[index0][0]];
    
    
    
        //loop over final join results
        for (int index1 = 0; index1 < table1.size(); ++index1) {
            if (1) {
                
                ++result;
                
                
            }
        }
    }

    cout << "Found " << result << " tuples.\n";
    
    
}

int main() { query(); }