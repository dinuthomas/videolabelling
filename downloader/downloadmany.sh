#!/bin/bash
downloadpath=../../../../input_video
configpath=./config.json
# Iterate the url array using for loop
for url in `cat url.txt`; do
 echo $url
 python videodownloader.py $url $downloadpath $configpath
done
