set -o errexit

add_id=1
catalog=msd_catalog_train.py
catalog_wid=msd_catalog_train_wid.py
#NOTE: convert does not actually pick this order; it picks the first navg+ncov features
navg=3
ncov=3
rel=trainingdata
input='train'
dataset=/sampa/home/bdmyers/escience/datasets/YearPredictionMSD_train.txt
binfile=/sampa/home/bdmyers/escience/datasets/YearPredictionMSD_train_6attr.bin
query=msd_train.myl
convert_home=../../c_test_environment

# without id
python create_scheme.py -a $navg -c $ncov --input=$input --no-id > $catalog
pushd $convert_home
python $convert_home/convert2bin.py -n $rel -c ../examples/naivebayes/$catalog
./$rel.convert $dataset 0 $add_id
popd
mv $dataset.bin $binfile

# add in id now that we added it
python create_scheme.py -a $navg -c $ncov --input=$input --no-id > $catalog_wid
../../scripts/myrial -c --catalog=$catalog_wid $query

