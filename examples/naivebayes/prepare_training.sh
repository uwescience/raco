set -o errexit

add_id=1
catalog=msd_catalog_train.py
catalog_wid=msd_catalog_train_wid.py
#NOTE: convert does not actually pick this order; it picks the first navg+ncov features
navg=4
ncov=4
rel=trainingdata
input='train'
dataset=/sampa/home/bdmyers/escience/datasets/millionsong/YearPredictionMSD_train.txt
#dataset=/sampa/home/bdmyers/escience/datasets/millionsong/YearPredictionMSD_train_small.txt
binfile=/sampa/home/bdmyers/escience/datasets/millionsong/YearPredictionMSD_train_8attr.txt
#binfile=/sampa/home/bdmyers/escience/datasets/millionsong/YearPredictionMSD_train_small_6attr.txt
queryin=naivebayes_train.myl
query=msd_train.myl
convert_home=../../c_test_environment

# without id
python create_scheme.py -a $navg -c $ncov --input=$input --no-id > $catalog
pushd $convert_home
python convert2bin.py -n $rel -c ../examples/naivebayes/$catalog
./$rel.convert $dataset 0 $add_id
popd
mv $dataset.bin $binfile.bin

# add in id now that we added it
python create_scheme.py -a $navg -c $ncov --input=$input > $catalog_wid
python generate_parse.py $(($navg + $ncov )) 1 > tmp.myl
cat tmp.myl $queryin > $query
../../scripts/myrial -c --emit=file --catalog=$catalog_wid $query


codef=`basename $query .myl`.cpp 
exef=grappa_`basename $query .myl`.exe
cp $codef $GRAPPA_HOME/applications/join/grappa_$codef
pushd $GRAPPA_HOME/build/Make+Release/applications/join
make -j $exef
popd

echo "--input_file_trainingdata=$binfile --output_file=$GRAPPA_HOME/build/Make+Release/applications/join/conditionals --relations=/"
