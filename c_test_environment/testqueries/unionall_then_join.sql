--unionall
select A.a from 
    (select a, b from T2
    union all
    select a, b from R2) A,
    S1 s
where s.a=A.a;
