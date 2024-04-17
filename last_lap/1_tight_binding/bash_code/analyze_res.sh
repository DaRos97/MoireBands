#!/bin/bash


echo "Import mafalda results"
./bash_code/maf.sh
echo "################################################## \nDFT vs fit tables"
python distance_dft.py
echo "################################################## \nOrbital content file"
python orbital_content.py
echo "################################################## \nPlots"
python Plot_final.py
