select MAX(b-c) from D3
UNION ALL
select MIN(c-b) from D3;
