\timing
set work_mem = '2000MB';
create table edges (src int, dst int);
copy edges from '/home/hyrkas/datalog_repo/datalogcompiler/c++/t_6200000' delimiter as ' ';
alter table edges add primary key (src,dst);
create index edges_src on edges(src);
create index edges_dst on edges(dst);
cluster edges using edges_src;
--non distinct two-paths...238 seconds???
--select count(*) from edges a, edges b where a.dst = b.src;
--distinct two-paths
--select count(*) from (select distinct a.src, b.dst from follows a, follows b where a.dst = b.src) c;
select count(*) from edges a, edges b, edges c where a.dst = b.src and b.dst = c.src and c.dst = a.src;
--check on how to allow postgres to use more main memory

/*
                                                 QUERY PLAN                                                 
------------------------------------------------------------------------------------------------------------
 Aggregate  (cost=419535528.69..419535528.70 rows=1 width=0)
   ->  Merge Join  (cost=406468191.01..419421515.02 rows=45605471 width=0)
         Merge Cond: ((b.dst = c.src) AND (a.src = c.dst))
         ->  Sort  (cost=405715849.88..409763969.03 rows=1619247661 width=8)
               Sort Key: b.dst, a.src
               ->  Merge Join  (cost=752341.13..25201323.44 rows=1619247661 width=8)
                     Merge Cond: (b.src = a.dst)
                     ->  Index Scan using edges_src on edges b  (cost=0.00..137760.52 rows=4532185 width=8)
                     ->  Materialize  (cost=752341.13..775002.06 rows=4532185 width=8)
                           ->  Sort  (cost=752341.13..763671.59 rows=4532185 width=8)
                                 Sort Key: a.dst
                                 ->  Seq Scan on edges a  (cost=0.00..65375.85 rows=4532185 width=8)
         ->  Materialize  (cost=752341.13..775002.06 rows=4532185 width=8)
               ->  Sort  (cost=752341.13..763671.59 rows=4532185 width=8)
                     Sort Key: c.src, c.dst
                     ->  Seq Scan on edges c  (cost=0.00..65375.85 rows=4532185 width=8)

*/
