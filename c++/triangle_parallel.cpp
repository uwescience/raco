//
//  query2.cpp
//  
//
//  Created by Jeremy Hyrkas on 4/17/13.
//
//

#include "triangle.h"
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <omp.h>

using namespace std;

void query(const char* filename) {
	
    //template for reading data
    vector<vector<int> > edges = vector<vector<int> >();
	ifstream edge_file(filename);
	int count = 0;
	vector<int> tmp = vector<int>();
	while (!edge_file.eof()) {
		int j;
		edge_file >> j;
		tmp.push_back(j);
		count++;
		if (count == 2) {
			count = 0;
			edges.push_back(tmp);
			tmp = vector<int>();
		}
	}
	edge_file.close();
    //end template

	cout << "done building vector\n";

	//hash on data[0]
	map<int, vector<vector<int> > > hash1;
	for (int i = 0; i < edges.size(); i++) {
		if (hash1.find(edges[i][0]) == hash1.end()) {
			hash1[edges[i][0]] = vector<vector<int> > ();
		}
		hash1[edges[i][0]].push_back(edges[i]);
	}

	cout << "done building map\n";

    omp_set_num_threads(32);

    count = 0;
    //loop over edges a,b
    #pragma omp parallel for reduction(+:count) schedule(dynamic,1)
    for (int i = 0; i < edges.size(); ++i) {
        int a = edges[i][0];
        if (hash1.find(edges[i][1]) == hash1.end()) {
            continue;
        }
        vector<vector<int> > matches1 = hash1[edges[i][1]];

        //loop over match b,c
        #pragma omp parallel for reduction(+:count) schedule(dynamic,1)
        for (int j = 0; j < matches1.size(); ++j) {
            int b = matches1[j][0];
            if (b <= a) {continue;}
            if (hash1.find(matches1[j][1]) == hash1.end()) {
                continue;
            }
            vector<vector<int> > matches2 = hash1[matches1[j][1]];

            //loop over matches c,d
            #pragma omp parallel for reduction(+:count) schedule(dynamic,1)
            for (int k = 0; k < matches2.size(); ++k) {
                int c = matches2[k][0];
                if (c <= b) { continue;}
                //select condition
                if (a == matches2[k][1]) {
                    //cout << "match: " << a << " " << b << " " << c << "\n";
                    ++count;
                }
            }
        }
    }

    cout << "Found " << count << "matches.\n";

}

int main(int argc, const char* argv[]) {
	query(argv[1]);
}
