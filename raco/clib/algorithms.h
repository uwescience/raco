
// function ptr syntax??
bool TwoPassSelect(condition(Tuple *), const Relation *input, Relation *output);

bool HashJoin(const Attribute &leftattr, const Attribute &rightattr, const Relation *left, const Relation *right, Relation *output);

bool Scan(string &name, const Catalog *catalog, Relation *output);
