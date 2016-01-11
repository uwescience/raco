select agg1.a, agg2.a
            from (select a, MIN(b) as mb from D2 group by a) agg1,
            (select a, MIN(b) as mb from D3 group by a) agg2
        where agg1.mb = agg2.mb;
