#!/bin/bash 
cd ${GRAPPA_HOME}
#./configure --gen=Make --mode=Release --cc=/sampa/share/distcc/gcc-4.7.2/bin/gcc --third-party=/sampa/share/grappa-third-party
./configure --gen=Make --mode=Release --cc=`which gcc` --third-party=/sampa/share/grappa-third-party/gcc-4.8.2
