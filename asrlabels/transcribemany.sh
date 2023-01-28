#!/bin/bash
urllist=/opt/iproject/videolabelling/downloader/url.txt
outfile=asrmany
# Iterate the url array using for loop
for url in `cat $urllist`; do
 echo $url
 python transcribe.py $url $outfile
done
