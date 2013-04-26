//
//  query.cpp
//  
//
//  Created by Jeremy Hyrkas on 4/14/13.
//
//

#include "query.h"
#include <map>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

void query() {
	vector<vector<int> > R = vector<vector<int> >();
	vector<vector<int> > T = vector<vector<int> >();
	
	//read R
	ifstream r_file("R");
	int count = 0;
	vector<int> tmp = vector<int>();
	while (!r_file.eof()) {
		int j;
		r_file >> j;
		tmp.push_back(j);
		count++;
		if (count == 4) {
			count = 0;
			R.push_back(tmp);
			tmp = vector<int>();
		}
	}
	r_file.close();
	
	//read T
	ifstream t_file("T");
	count = 0;
	tmp = vector<int>();
	while (!t_file.eof()) {
		int j;
		t_file >> j;
		tmp.push_back(j);
		count++;
		if (count == 4) {
			count = 0;
			T.push_back(tmp);
			tmp = vector<int>();
		}
	}
	t_file.close();
	
	//hash on T
	//condition R.c = T.a
	map<int, vector<vector<int> > > hash;
	for (int i = 0; i < T.size(); i++) {
		//cout << T[i][0];
		//cout << endl;
		if (hash.find(T[i][0]) == hash.end()) {
			hash[T[i][0]] = vector<vector<int> > ();
		}
		hash[T[i][0]].push_back(T[i]);
	}
	
	//search R
	for (int i = 0; i < R.size(); i++) {
		if (hash.find(R[i][2]) != hash.end()) {
			cout << "match ";
			cout << R[i][2];
			cout << endl;
		}
	}
}

int main() {
	query();
}