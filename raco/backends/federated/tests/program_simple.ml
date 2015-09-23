abc = scan(abc);
a = [from abc emit value as a_val];
a1 = [from abc emit value as a1_val];
b = [from a, a1 where a.a_val + 1 = a1.a1_val emit a1.a1_val];
store(b, filtered_array);
