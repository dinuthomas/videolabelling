#!/bin/bash
source /opt/capstone/capstone/bin/activate
sh infervideos.sh
python ytbtokenmap.py vocabulary.csv data/results.csv data/mappedresults