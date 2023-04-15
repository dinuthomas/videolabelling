from wikidata_dump import getwikiEntities, prepareTrainData
from urllib import request  
import argparse
import pandas as pd
import csv
import ast
import os
import shutil
from urllib.request import urlopen
from shutil import copyfileobj
from os import path

def readline(file):
    with open(file, newline='') as csvfile:
        wiki = csv.reader(csvfile, delimiter='\t')
        for row in wiki:
            yield row

def get_filtered_entities(filtered_wikidata_path, lang_list, wikioutfile="wikientities_all.out"):
    cnt=0
    heads=["entity_ids","entity_lang","entity_txt","sitelinks"]
    lang = lang_list

    with open(filtered_wikidata_path, 'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(heads) 
        i = 0

        for line in readline(wikioutfile):
            try: 
                res = list(line)
                i += 1 

                for j in range(len(lang)):
                    if len(res[3+2*j]) == 0:
                        continue
                    dd = ast.literal_eval((res[3+2*j]))

                    ent = [(res[2+2*j])] + dd


                    for entity in ent:
                        data = [res[0],lang[j],entity,res[-1]]
                        cnt += 1
                        writer.writerow(data)
            except:
                print(line)
                continue

        if i%500000==0:
            print("i==",i)
                
    print("total record==",cnt)


def data_saved(df, lang, file_name):
    print("started saving file: ",lang)
    df_new=df[df['entity_lang']==str(lang)]
    df_new.to_csv(file_name)


langs = ['en','zh','es','ar']
def data_save_diff_file(langs, filtered_wikidata_path):
    df = pd.read_csv(filtered_wikidata_path, index_col=False)
    print("file_loaded")
    for lang in langs:
        file_name=filtered_wikidata_path.split('.')[0]+'_'+str(lang)+'.csv'
        data_saved(df, lang, file_name)
        print("data saved ",str(lang))

def groupby_text(filtered_file_name,groupby_file_path,lang):
    df = pd.read_csv(filtered_file_name, index_col=False)  
    print("data loaded")
    df['entity_txt']= df['entity_txt'].str.lower()
    df_new = df[['entity_ids','entity_lang','entity_txt','sitelinks']]
    print(" grouping started ")
    df_grouped = df_new.groupby(['entity_lang','entity_txt']).agg(lambda x: list(x)).reset_index()
    print("grouping done and now saving...")
    df_grouped.to_csv(groupby_file_path)
    print(" grouping done and saved :", lang)


def groupby_entity(lang_list, groupby_entity_dir, filtered_wikidata_path):
    if os.path.exists(groupby_entity_dir):
        shutil.rmtree(groupby_entity_dir)
    os.makedirs(groupby_entity_dir)

    for lang in lang_list:
        filtered_file_name = filtered_wikidata_path.split('.')[0]+'_'+str(lang)+'.csv'
        groupby_file_name = groupby_entity_dir.split('.')[0]+'_'+str(lang)+'.csv'
        groupby_file_path = os.path.join(groupby_entity_dir,groupby_file_name)
        groupby_text(filtered_file_name,groupby_file_path,lang)
    print("all groupby data saved")

 
#--------agruments (wiki_url path, filtered_wikidata_path)-------
parser = argparse.ArgumentParser()
parser.add_argument("url", help="wiki_url path", type=str)
parser.add_argument("output_path", help="groupby_entity_savepath_csv", type=str)

args = parser.parse_args()
url_path=args.url
out_path=args.output_path
lang_list = ["en"]

#-----download the wiki json file--------
print("----downloading started from url----")
if False == path.exists("latest-all.json.bz2"):
    response = request.urlretrieve(url_path, "latest-all.json.bz2")  
    with urlopen(url_path) as in_stream, open('latest-all.json.bz2', 'wb') as out_file:
        copyfileobj(in_stream, out_file)
else:
    print("the wikidump s already downloaded")
print("----downloading completed----")

#----- Extract the relevant properties of entities from the wiki data ----------------
print("----started extracting relevant entities from the wikidata----")
if False == path.exists("wikientities_all.out"):
    getwikiEntities(r"latest-all.json.bz2", r"wikientities_all.out", lang_list) 
else:
    print("the wikientities_all is already processed")

print("----extraction completed----")

if False == path.exists("traindata.out"):
    prepareTrainData(r"latest-all.json.bz2", r"traindata.out", lang_list) 
else:
    print("the training data is already processed")

print("----training data completed----")

'''
#-----filter the wiki data ----------------
if False == path.exists("filtered_wikidata_all.csv"):
    get_filtered_entities("filtered_wikidata_all.csv",
    lang_list, \
    wikioutfile="wikientities_all.out") 
else:
    print("the filtered_wikidata_all is already processed")


print("----filtering done successfully and stored----")

#data_save_diff_file(lang_list,filtered_wikidata_path="filtered_wikidata_all.csv")

#-----group by the entity from filtered wiki data ----------------
print("----started grouping the filtered wikidata----")
groupby_entity(lang_list, groupby_entity_dir=out_path,filtered_wikidata_path="filtered_wikidata_all.csv")
#groupby_entity( groupby_entity_dir=out_path,filtered_wikidata_path="filtered_wikidata_all.csv")
print("----groupby done successfully and stored in csv file----")
'''







        
 







        
 





        
 





        
 


