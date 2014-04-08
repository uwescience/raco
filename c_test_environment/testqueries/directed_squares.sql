select r.a, s.a, t.a, z.a from R2 r, S2 s, T2 t, R3 z where r.b=s.a and s.b=t.a and t.b=z.a and z.b=r.a;
