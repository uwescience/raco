query=$1
name=$2

cdir=`cd ..; pwd`
cappsrcdir=$cdir/c_test_environment
cbuilddir=$cdir/c_test_environment
cappbuilddir=$gbuilddir/applications/join

pushd $cbuilddir
if [ ! -f R1 ]; then
  echo "GENERATING TEST DATA (first time)"
  python generate_test_relations.py
fi
popd
  

echo "GENERATING QUERY CODE"
PYTHONPATH=.. python clog.py "$query" $name 2> log.rb
mv $name.cpp $cappsrcdir

echo "COMPILING QUERY CODE"
cd $cbuilddir; make $name.exe; echo "RUNNING QUERY CODE"; ./$name.exe

