#include "A.h"

void query () {

    //scan R
    vector<vector<int> > R = vector<vector<int> >();
    ifstream f0("R");
    int count0 = 0;
    vector<int> tmp_vector0 = vector<int>();
    while (!f0.eof()) {
    	int j;
    	f0 >> j;
    	tmp_vector0.push_back(j);
    	count0++;
    	if (count0 == 2) {
    		count0 = 0;
    		R.push_back(tmp_vector0);
    		tmp_vector0 = vector<int>();
    	}
    }
    f0.close();
    
    
    
    
    //scan S
    vector<vector<int> > S = vector<vector<int> >();
    ifstream f1("S");
    int count1 = 0;
    vector<int> tmp_vector1 = vector<int>();
    while (!f1.eof()) {
    	int j;
    	f1 >> j;
    	tmp_vector1.push_back(j);
    	count1++;
    	if (count1 == 2) {
    		count1 = 0;
    		S.push_back(tmp_vector1);
    		tmp_vector1 = vector<int>();
    	}
    }
    f1.close();
    
    
    
    
    //scan T
    vector<vector<int> > T = vector<vector<int> >();
    ifstream f2("T");
    int count2 = 0;
    vector<int> tmp_vector2 = vector<int>();
    while (!f2.eof()) {
    	int j;
    	f2 >> j;
    	tmp_vector2.push_back(j);
    	count2++;
    	if (count2 == 2) {
    		count2 = 0;
    		T.push_back(tmp_vector2);
    		tmp_vector2 = vector<int>();
    	}
    }
    f2.close();
    
    
    
    
    //scan U
    vector<vector<int> > U = vector<vector<int> >();
    ifstream f3("U");
    int count3 = 0;
    vector<int> tmp_vector3 = vector<int>();
    while (!f3.eof()) {
    	int j;
    	f3 >> j;
    	tmp_vector3.push_back(j);
    	count3++;
    	if (count3 == 2) {
    		count3 = 0;
    		U.push_back(tmp_vector3);
    		tmp_vector3 = vector<int>();
    	}
    }
    f3.close();
    
    
    
    
    //hash S
    map<int, vector<vector<int> > > S0_hash;
    for (int i = 0; i < S.size(); i++) {
    	if (S0_hash.find(S[i][0]) == S0_hash.end()) {
    		S0_hash[S[i][0]] = vector<vector<int> > ();
    	}
    	S0_hash[S[i][0]].push_back(S[i]);
    }
    
    
    
    
    //hash T
    map<int, vector<vector<int> > > T0_hash;
    for (int i = 0; i < T.size(); i++) {
    	if (T0_hash.find(T[i][0]) == T0_hash.end()) {
    		T0_hash[T[i][0]] = vector<vector<int> > ();
    	}
    	T0_hash[T[i][0]].push_back(T[i]);
    }
    
    
    
    
    //hash U
    map<int, vector<vector<int> > > U0_hash;
    for (int i = 0; i < U.size(); i++) {
    	if (U0_hash.find(U[i][0]) == U0_hash.end()) {
    		U0_hash[U[i][0]] = vector<vector<int> > ();
    	}
    	U0_hash[U[i][0]].push_back(U[i]);
    }
    
    
    
    
}

int main() { query(); }