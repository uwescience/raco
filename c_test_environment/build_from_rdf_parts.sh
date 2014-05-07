#!/bin/bash
base=$1
last=`ls -l $base.part.* | wc -l`

touch $base.str
for i in `seq 0 $last`;
do
    ./rdfprint $base.part.$i >>$base.str
done
