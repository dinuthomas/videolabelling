# -*- coding: utf-8 -*-

import bz2
import json
from random import random
import ast


def wikidata(filename):
    '''generator for getting one raw at a time'''
    with bz2.open(filename, mode='rt') as f:
        f.read(2) # skip first two bytes: "{\n"
        for line in f:
            try:
                yield json.loads(line.rstrip(',\n'))
            except json.decoder.JSONDecodeError:
                continue


def getTypes(record):
    '''get [instance of] relation id list for given entity'''
    typeList = []
    try:
        if record['claims']['P31'] is not None:
            for i in record['claims']['P31']:
                try:
                    typeList.append(i['mainsnak']['datavalue']['value']['id'])
                except KeyError:
                    pass
    except KeyError as ke:
        pass
    return typeList


def getLangValue(element,lang):
    '''get language value for a element'''
    val = ""
    try:
        if element is not None and element[lang] is not None:
            val = element[lang]['value']
    except KeyError as ke:
        pass
    return val

def getAliases(element, lang):
    ''' get aliases for the element '''
    val = ""
    try:
        if element is not None and element[lang] is not None:
            alias_dict = ast.literal_eval(str(element[lang]))
            val = [x['value'] for x in alias_dict]
            
    except KeyError as ke:
        pass
    return val

def getwikiEntities(wikifile, fileout, lang_list):
    ''' get id, name , [typeof] ids list in tsv  '''
    i = 0
    nexti = 0

    with open(fileout, 'w', encoding='utf-8') as fout:
        for record in wikidata(wikifile):
            #print("1: " + str(record))           
            

            sitelinks = 0
            if i == 10:
                break
            if i % 50000 == 0:
                print(i)
            if True: #i == nexti :
                line = ""
                line += record['id']
                print("2: " +record['id'])
                line += "\t"

                try:
                    line += str(getTypes(record))
                    line += "\t"
                    for lang in lang_list:
                        val = getLangValue(record['labels'], lang)  
                        print("3: " +val)
                        line += val
                        line += "\t"
                        val = getAliases(record['aliases'], lang) 
                        print("4: " +str(val) )
                        line += str(val)
                        line += "\t"
                        line += str(line)
                    if 'sitelinks' in record.keys():
                        sitelink = len(record['sitelinks'].keys())
                        print("5: " +str(sitelink))
                    line += str(sitelink)
                    line += "\t"
                    line += "\n"
                    
                    fout.write(line)

                except KeyError as ke:
                    print("##ERROR>>" + str(ke))
                    pass
            i= i+1

def prepareTrainData(wikifile, fileout, lang_list):
    ''' get id, name , [typeof] ids list in tsv  '''
    i = 0
    nexti = 0

    with open(fileout, 'w', encoding='utf-8') as fout:
        for record in wikidata(wikifile):
            #print("1: " + str(record))           
            

            sitelinks = 0
            if i % 50000 == 0:
                print(i)
            if True: #i == nexti :
                line = ""
                wiki_id = record['id']

                try:
                    rec_type = getTypes(record)
                    for lang in lang_list:
                        line += wiki_id
                        line += "\t"
                        label = getLangValue(record['labels'], lang).lower()  
                        line += label
                        line += "\t"
                        line += label
                        line += "\n"

                        line += wiki_id
                        line += "\t"
                        line += wiki_id
                        line += "\t"
                        line += label
                        line += "\n"
                        for item in rec_type:
                            line += wiki_id
                            line += "\t"
                            line += item
                            line += "\t"
                            line += label
                            line += "\n"

                        alias = getAliases(record['aliases'], lang)
                        for item in alias:
                            line += wiki_id
                            line += "\t"
                            line += item.lower()
                            line += "\t"
                            line += label
                            line += "\n"
                    fout.write(line)

                except KeyError as ke:
                    print("##ERROR>>" + str(ke))
                    pass
            i= i+1


