a = scan(AAA);
b = scan(BBB);
m = select a_i, b_j, sum(a_v*b_v) from a, b where a_j = b_i;
store(m,mult);
