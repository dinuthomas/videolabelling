#!/bin/bash

source /opt/iproject/ner/bin/activate
python tokenmapper.py ../../asrlabels/asrmany_rake ../wikigraph/graph/ wikiidmap_rake
