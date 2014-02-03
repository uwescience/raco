query=$1
name=grappa_$2

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
PYTHONPATH=.. python grappalog.py "$query" $name 2> log.rb
cp $name.cpp $gappsrcdir

echo "COMPILING QUERY CODE"
#TODO: make this not so new target dependent. Easy way is have a set of default targets that can be recycled
cd $gdir; ./configure --gen=Make --mode=Release --cc=/sampa/share/distcc/gcc-4.7.2/bin/gcc --third-party=/sampa/share/grappa-third-party
cd $gbuilddir; bin/distcc_make -j 24; cd $gappbuilddir; ../../bin/distcc_make $name.exe; echo "RUNNING QUERY CODE"; ../../bin/grappa_srun --ppn=4 --nnode=4 -f -- $name.exe
#cd $gappbuilddir; ../../bin/distcc_make $name.exe; echo "RUNNING QUERY CODE"; ../../bin/grappa_srun --ppn=4 --nnode=4 -f -- $name.exe

