--unionall
select A1.a from 
(select a, b from T2
union all 
select a, b from R2) A1,
(select a, b from T2
union all 
select a, b from R2) A2
where A1.a=A2.a;

