
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

# without id
python create_scheme.py -a $navg -c $ncov --input=$input --no-id > $catalog
python ../../c_test_environment/convert2bin.py -n $rel -c $catalog
./$rel.convert $dataset 0 $add_id
mv $dataset.bin $binfile

# add in id now that we added it
python create_scheme.py -a $navg -c $ncov --input=$input --no-id > $catalog_wid
../../scripts/myrial -c --catalog=$catalog_wid $query

