touch "ingest_all_cosmo.sh"
for VARIABLE in [SNAPSHOTS]
do
	touch "ingest_cosmo$VARIABLE.json"
	echo -e "{ \n\
    	\"grpFilename\": \"[PATH_TO_SCRIPTS]$VARIABLE.amiga.grp\",\n\
    	\"iorderFilename\": \"[PATH_TO_SCRIPTS]$VARIABLE.iord_2\",\n\
    	\"relationKey\": {\n\
        	\"programName\": \"[SIMULATION_NAME]\",\n\
        	\"relationName\": \"cosmo$VARIABLE\",\n\
        	\"userName\": \"public\"\n\
    	},\n\
    	\"tipsyFilename\": \"[PATH_TO_SCRIPTS]$VARIABLE\"\n\
	}" > "ingest_cosmo$VARIABLE.json"
	echo -e "curl -i -XPOST https://rest.myria.cs.washington.edu:1776/dataset/tipsy -H \"Content-type: application/json\"  -d @./ingest_cosmo${VARIABLE}.json\n" >> "ingest_all_cosmo.sh"
done