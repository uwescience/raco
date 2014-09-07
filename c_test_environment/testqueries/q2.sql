select T1.a as inproc, T2.c as author, T3.c as booktitle, T4.c as title, T5.c as proc, T6.c as ee, T7.c as page, T8.c as url, T9.c as yr
from R3 T1,
            R3 T2,
            R3 T3,
            R3 T4,
            R3 T5,
            R3 T6,
            R3 T7,
            R3 T8,
            R3 T9 
WHERE T1.a=T2.a
and T2.a=T3.a
and T3.a=T4.a
and T4.a=T5.a
and T5.a=T6.a
and T6.a=T7.a
and T7.a=T8.a
and T8.a=T9.a
and T1.b = 1 and T1.c > 5
and T2.b = 1
and T3.b = 1
and T4.b = 1
and T5.b = 1
and T6.b = 1
and T7.b = 1
and T8.b = 1
and T9.b = 1;
