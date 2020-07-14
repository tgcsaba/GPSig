DATA_DIR="datasets/"

wget https://www.dropbox.com/sh/jqy6519ogx0ool3/AADyDdCPTZCj3QQbGXrYNjiua/AllDatasets.zip?dl=1 -O ${DATA_DIR}/AllDatasets.zip
unzip ${DATA_DIR}/AllDatasets.zip -d ${DATA_DIR}
rm ${DATA_DIR}/AllDatasets.zip