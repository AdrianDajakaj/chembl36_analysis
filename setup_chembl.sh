#!/bin/bash
set -e

mkdir -p downloads

if [ ! -f downloads/chembl_36_postgresql.tar.gz ]; then
    echo "Downloading ChEMBL 36 database..."
    curl -o downloads/chembl_36_postgresql.tar.gz https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_postgresql.tar.gz
else
    echo "Database file already exists in the downloads folder."
fi

echo "Extracting..."
tar -xvzf downloads/chembl_36_postgresql.tar.gz -C downloads/

echo "Waiting for the container to be ready..."
until docker exec chembl_container pg_isready -U admin -d chembl_36; do
    echo "Container not ready yet, retrying in 5s..."
    sleep 5
done

echo "Importing data into PostgreSQL (this may take a while)..."
docker exec -i chembl_container pg_restore -U admin -d chembl_36 -v < downloads/chembl_36/chembl_36_postgresql/chembl_36_postgresql.dmp

echo "Import complete."