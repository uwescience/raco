//
//  query2.cpp
//  
//
//  Created by Jeremy Hyrkas on 4/17/13.
//
//

#include "query2.h"
#include <map>
#include <vector>
#include <iostream>
#include <fstream>

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
	
	//hash on data[0]
	map<int, vector<vector<int> > > hash1;
	for (int i = 0; i < edges.size(); i++) {
		if (hash1.find(edges[i][0]) == hash1.end()) {
			hash1[edges[i][0]] = vector<vector<int> > ();
		}
		hash1[edges[i][0]].push_back(edges[i]);
	}

    //loop over edges a,b
    for (int i = 0; i < edges.size(); ++i) {
        int a = edges[i][0];
        if (hash1.find(edges[i][1]) == hash1.end()) {
            continue;
        }
        vector<vector<int> > matches1 = hash1[edges[i][1]];

        //loop over match b,c
        for (int j = 0; j < matches1.size(); ++j) {
            int b = matches1[j][0];
            if (hash1.find(matches1[j][1]) == hash1.end()) {
                continue;
            }
            vector<vector<int> > matches2 = hash1[matches1[j][1]];

            //loop over matches c,d
            for (int k = 0; k < matches2.size(); ++k) {
                int c = matches2[k][0];

                //select condition
                if (a == matches2[k][1] && a < b && b < c) {
                    cout << "match: " << a << " " << b << " " << c << "\n";
                }
            }
        }
    }


    /*old search
	//search for triangle: T(a,b,c) :- edges(a,b),edges(b,c),edges(c,a),a<b<c
	//for all edges(a,b)
	for (map<int,vector<vector<int> > >::iterator iter1 = hash1.begin(); iter1 != hash1.end(); ++iter1) {
		int a = iter1->first;
		if (hash1.find(a) == hash1.end()) {
			continue;
		}
		for (int i = 0; i < hash1[a].size(); ++i) {
			//for all edges (b,c)
			int b = hash1[a][i][1];
			if (hash1.find(b) == hash1.end()) {
				continue;
			}
			//for all edges (c,d)
			for (int j = 0; j < hash1[b].size(); ++j) {
				int c = hash1[b][j][1];
				if (hash1.find(c) == hash1.end()) {
					continue;
				}
				for (int k = 0; k < hash1[c].size(); ++k) {
					int d = hash1[c][k][1];
					//if d=a and a<b<c
					if (d == a && a < b && b < c) {
						cout << "match: " << a << " " << b << " " << c << "\n";
					}
				}
			}
		}
	}
    */

}

int main(int argc, const char* argv[]) {
	query(argv[1]);
}
