touch "ingest_all_cosmo.sh"
for VARIABLE in 0045 0054 0072 0102 0128 0133 0155 0182 0219 0256 0269 0302 0342 0384 0390 0451 0512 0529 0632 0640 0768 0772 0896 0970 1024 1152 1265 1280 1408 1536 1664 1737 1792 1920 2048 2085 2176 2304 2432 2552 2560
do
	touch "ingest_cosmo$VARIABLE.json"
	echo -e "{ \n\
    	\"grpFilename\": \"/disk3/jortiz16/MichaelDatasets/Romulus/romulus8.256gst2.bwBH/romulus8.256gst2.bwBH.00$VARIABLE.amiga.grp\",\n\
    	\"iorderFilename\": \"/disk3/jortiz16/MichaelDatasets/Romulus/romulus8.256gst2.bwBH/romulus8.256gst2.bwBH.00$VARIABLE.iord_2\",\n\
    	\"relationKey\": {\n\
        	\"programName\": \"romulus8\",\n\
        	\"relationName\": \"cosmo$VARIABLE\",\n\
        	\"userName\": \"public\"\n\
    	},\n\
    	\"tipsyFilename\": \"/disk3/jortiz16/MichaelDatasets/Romulus/romulus8.256gst2.bwBH/romulus8.256gst2.bwBH.00$VARIABLE\"\n\
	}" > "ingest_cosmo$VARIABLE.json"
	echo -e "curl -i -XPOST https://rest.myria.cs.washington.edu:1776/dataset/tipsy -H \"Content-type: application/json\"  -d @./ingest_cosmo${VARIABLE}.json\n" >> "ingest_all_cosmo.sh"
done