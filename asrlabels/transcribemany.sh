#!/bin/bash
source /opt/iproject/py39/bin/activate
. /opt/iproject/py39/bin/activate
outfile=asrmany
argcount=$#
homepath=/opt/iproject/videolabelling/asrlabels/
cd /opt/iproject/videolabelling/asrlabels/
echo "count of arguments = $argcount"
if [ $argcount != 0 ];
then
 echo 'url is in the argument'
 url=$1
 python /opt/iproject/videolabelling/asrlabels/transcribed.py $url $outfile $homepath
else
 echo 'take urls present locally'
 urllist=/opt/iproject/videolabelling/downloader/url.txt
 # Iterate the url array using for loop
 for url in `cat $urllist`; do
  echo $url
  python /opt/iproject/videolabelling/asrlabels/transcribed.py $url $outfile $homepath
 done
fi	
