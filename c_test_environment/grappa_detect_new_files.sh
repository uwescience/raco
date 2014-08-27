#!/bin/bash 
cd ${GRAPPA_HOME}
./configure --gen=Make --mode=Release --c=/sampa/share/gcc-4.8.2/rtf/bin/gcc \
 --third-party=/sampa/share/grappa-third-party/gcc-4.8.2