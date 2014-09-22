# created by msd_train.myl:
# conditionals (scheme)
# conditionals.bin (data)

add_id=1
catalog=msd_catalog_test.py
catalog_wid=msd_catalog_test_wid.py
navg=3
ncov=3
rel=testdata
input='test'
dataset=/sampa/home/bdmyers/escience/datasets/YearPredictionMSD_test.txt
binfile=/sampa/home/bdmyers/escience/datasets/YearPredictionMSD_test_6attr.bin
query=msd_classfy.myl

# without id
python create_scheme.py -a $navg -c $ncov --input=$input --no-id > $catalog
python ../../c_test_environment/convert2bin.py -n $rel -c $catalog
./$rel.convert $dataset 0 $add_id

# add in id now that we added it
python create_scheme.py -a $navg -c $ncov --input=$input --no-id > $catalog_wid
../../scripts/myrial -c --catalog=$catalog_wid $query

