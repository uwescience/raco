set -o errexit

# created by msd_train.myl:
# conditionals (scheme)
# conditionals.bin (data)

add_id=1
catalog=msd_catalog_test.py
catalog_wid=msd_catalog_test_wid.py
catalog_all=msd_catalog_test_all.py
navg=4
ncov=4
rel=testdata
input='test'
dataset=/sampa/home/bdmyers/escience/datasets/millionsong/YearPredictionMSD_test.txt
#dataset=/sampa/home/bdmyers/escience/datasets/millionsong/YearPredictionMSD_test_small.txt
binfile=/sampa/home/bdmyers/escience/datasets/millionsong/YearPredictionMSD_test_8attr.txt
#binfile=/sampa/home/bdmyers/escience/datasets/millionsong/YearPredictionMSD_test_small_6attr.txt
queryin=naivebayes_classify.myl
query=msd_classify.myl
convert_home=../../c_test_environment

# without id
python create_scheme.py -a $navg -c $ncov --input=$input --no-id --no-y > $catalog
pushd $convert_home
python convert2bin.py -n $rel -c ../examples/naivebayes/$catalog
./$rel.convert $dataset 0 $add_id
popd
mv $dataset.bin $binfile.bin

# add in id now that we added it
python create_scheme.py -a $navg -c $ncov --input=$input --no-y > $catalog_wid
./cat_scheme $GRAPPA_HOME/build/Make+Release/applications/join/conditionals $catalog_wid > $catalog_all
python generate_parse.py $(($navg + $ncov)) 0 > tmp.myl
cat tmp.myl $queryin > $query
../../scripts/myrial --emit=console -c --catalog=$catalog_all $query


codef=`basename $query .myl`.cpp 
exef=grappa_`basename $query .myl`.exe
scp $codef pal:~/grappa-nb/applications/join/grappa_$codef
cp $codef $GRAPPA_HOME/applications/join/grappa_$codef
pushd $GRAPPA_HOME/build/Make+Release/applications/join
make -j $exef
popd

echo "--input_file_conditionals=$GRAPPA_HOME/build/Make+Release/applications/join/conditionals --output_file=$GRAPPA_HOME/build/Make+Release/applications/join/classified --input_file_testdata=$binfile --relations=/"
