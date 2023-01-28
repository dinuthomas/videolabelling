#!/bin/bash
urllist=/opt/iproject/videolabelling/downloader/url.txt
outfile=asrmany_rake
# Iterate the url array using for loop
for url in `cat $urllist`; do
 echo $url
 python transcribe_rake.py $url $outfile
done
