#include "test.h"

void query () {
vector<vector<int> > R = vector<vector<int> >();
ifstream f1("R");
int count1 = 0;
vector<int> tmp_vector1 = vector<int>();
while (!f1.eof()) {
	int j;
	f1 >> j;
	tmp_vector1.push_back(j);
	count1++;
	if (count1 == 4) {
		count1 = 0;
		R.push_back(tmp_vector1);
		tmp_vector1 = vector<int>();
	}
}
f1.close();

vector<vector<int> > T = vector<vector<int> >();
ifstream f2("T");
int count2 = 0;
vector<int> tmp_vector2 = vector<int>();
while (!f2.eof()) {
	int j;
	f2 >> j;
	tmp_vector2.push_back(j);
	count2++;
	if (count2 == 4) {
		count2 = 0;
		T.push_back(tmp_vector2);
		tmp_vector2 = vector<int>();
	}
}
f2.close();

}

int main() { query(); }