a = scan(A);
b = scan(B);
m = [from a, b where a_i = b_j emit a_v];
store(m,join_result);
