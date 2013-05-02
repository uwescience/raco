#include "A.h"

void query () {

    //scan T
    vector<vector<int> > T = vector<vector<int> >();
    ifstream f0("T");
    int count0 = 0;
    vector<int> tmp_vector0 = vector<int>();
    while (!f0.eof()) {
    	int j;
    	f0 >> j;
    	tmp_vector0.push_back(j);
    	count0++;
    	if (count0 == 2) {
    		count0 = 0;
    		T.push_back(tmp_vector0);
    		tmp_vector0 = vector<int>();
    	}
    }
    f0.close();
    
    
    
    
    //scan U
    vector<vector<int> > U = vector<vector<int> >();
    ifstream f1("U");
    int count1 = 0;
    vector<int> tmp_vector1 = vector<int>();
    while (!f1.eof()) {
    	int j;
    	f1 >> j;
    	tmp_vector1.push_back(j);
    	count1++;
    	if (count1 == 2) {
    		count1 = 0;
    		U.push_back(tmp_vector1);
    		tmp_vector1 = vector<int>();
    	}
    }
    f1.close();
    
    
    
    
    //scan S
    vector<vector<int> > S = vector<vector<int> >();
    ifstream f2("S");
    int count2 = 0;
    vector<int> tmp_vector2 = vector<int>();
    while (!f2.eof()) {
    	int j;
    	f2 >> j;
    	tmp_vector2.push_back(j);
    	count2++;
    	if (count2 == 2) {
    		count2 = 0;
    		S.push_back(tmp_vector2);
    		tmp_vector2 = vector<int>();
    	}
    }
    f2.close();
    
    
    
    
    //scan R
    vector<vector<int> > R = vector<vector<int> >();
    ifstream f3("R");
    int count3 = 0;
    vector<int> tmp_vector3 = vector<int>();
    while (!f3.eof()) {
    	int j;
    	f3 >> j;
    	tmp_vector3.push_back(j);
    	count3++;
    	if (count3 == 2) {
    		count3 = 0;
    		R.push_back(tmp_vector3);
    		tmp_vector3 = vector<int>();
    	}
    }
    f3.close();
    
    
    
    
    //hash U
    map<int, vector<vector<int> > > U0_hash;
    for (int i = 0; i < U.size(); i++) {
    	if (U0_hash.find(U[i][0]) == U0_hash.end()) {
    		U0_hash[U[i][0]] = vector<vector<int> > ();
    	}
    	U0_hash[U[i][0]].push_back(U[i]);
    }
    
    
    
    
    //hash S
    map<int, vector<vector<int> > > S1_hash;
    for (int i = 0; i < S.size(); i++) {
    	if (S1_hash.find(S[i][1]) == S1_hash.end()) {
    		S1_hash[S[i][1]] = vector<vector<int> > ();
    	}
    	S1_hash[S[i][1]].push_back(S[i]);
    }
    
    
    
    
    //hash R
    map<int, vector<vector<int> > > R1_hash;
    for (int i = 0; i < R.size(); i++) {
    	if (R1_hash.find(R[i][1]) == R1_hash.end()) {
    		R1_hash[R[i][1]] = vector<vector<int> > ();
    	}
    	R1_hash[R[i][1]].push_back(R[i]);
    }
    
    
    
    
    //loop over T
    for (int index0 = 0; index0 < T.size(); ++index0) {
        if (U0_hash.find(T[index0][1]) == U0_hash.end()) {
            continue;
        }
        vector<vector<int> > table1 = U0_hash[T[index0][1]];
    
    
    
        //loop over table1
        for (int index1 = 0; index1 < table1.size(); ++index1) {
            if (S1_hash.find(table1[index1][1]) == S1_hash.end()) {
                continue;
            }
            vector<vector<int> > table2 = S1_hash[table1[index1][1]];
        
        
        
            //loop over table2
            for (int index2 = 0; index2 < table2.size(); ++index2) {
                if (R1_hash.find(table2[index2][0]) == R1_hash.end()) {
                    continue;
                }
                vector<vector<int> > table3 = R1_hash[table2[index2][0]];
            
            
            
            }
        }
    }

}

int main() { query(); }