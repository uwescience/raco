#!/bin/sh
query=$1
name=$2
plan=$3


cappbuilddir=`cd ../c_test_environment; pwd`
gdir=$GRAPPA_HOME
gappsrcdir=$gdir/applications/join
gbuilddir=$gdir/build/Make+Release
gappbuilddir=$gbuilddir/applications/join

pushd $gappbuilddir
if [ ! -f R1 ]; then
  echo "GENERATING TEST DATA (first time)"
  python $cappbuilddir/generate_test_relations.py
fi
popd

echo "GENERATING QUERY CODE"
PYTHONPATH=.. python grappalog.py "$query" $name $plan 2> log.rb
# get file name assuming it is most recent cpp file
fullname=`ls -lt *cpp | head -n1 |awk '{gsub(/ +/, " ");print}' | cut -d' ' -f9 | cut -d'.' -f1`
cp $fullname.cpp $gappsrcdir

echo "COMPILING QUERY CODE"
#TODO: make this not so new target dependent. Easy way is have a set of default targets that can be recycled
cd $gdir; ./configure --gen=Make --mode=Release --cc=/sampa/share/distcc/gcc-4.7.2/bin/gcc --third-party=/sampa/share/grappa-third-party
cd $gbuilddir; bin/distcc_make -j 24; cd $gappbuilddir; ../../bin/distcc_make -j24 $fullname.exe; echo "RUNNING QUERY CODE"; ../../bin/grappa_srun --ppn=4 --nnode=4 -f -- $fullname.exe
#cd $gappbuilddir; ../../bin/distcc_make $fullname.exe; echo "RUNNING QUERY CODE"; ../../bin/grappa_srun --ppn=4 --nnode=4 -f -- $fullname.exe

