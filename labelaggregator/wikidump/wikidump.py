from wikidata_dump import getwikiEntities
from urllib import request  
import argparse
import pandas as pd
import csv
import ast
import os
import shutil
from urllib.request import urlopen
from shutil import copyfileobj

#------filter from wiki_dump file----------------
def get_filtered_entities(filtered_wikidata_path, wikioutfile="wikientities_all.out"):
    file1 = open(wikioutfile, 'r',encoding='utf8')
    Lines = file1.readlines()
    cnt=0
    heads=["entity_ids","entity_lang","entity_txt","sitelinks"]
    lang = ['en','zh','es','ar']
    with open(filtered_wikidata_path, 'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(heads)        
        for i,line in enumerate(Lines):
            #print('line== '+line)
            res = line.split('\t')   
            '''print('res== ') 
            print(res)   '''    
            res1=res[2].split(" ") 
            '''print('res1== ') 
            print(res1) '''
            if len(res1)>5:
                continue

            for j in range(4):
                if len(res[3+2*j]) == 0:
                    continue
                dd = ast.literal_eval(res[3+2*j])
                '''print('dd == ')
                print(dd)'''
                ent = [str(res[2+2*j])] + dd
                '''print('ent == ')
                print(ent)'''

                for entity in ent:
                    data = [str(res[0]),lang[j],entity,res[10]]
                    '''print('data == ')
                    print(data)'''
                    writer.writerow(data)
                
            '''if i>2:
                break'''

                #break             
            if i%50000==0:
                print("i==",i)
                #break 
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


def groupby_entity(groupby_entity_dir, filtered_wikidata_path):
    if os.path.exists(groupby_entity_dir):
        shutil.rmtree(groupby_entity_dir)
    os.makedirs(groupby_entity_dir)

    for lang in langs:
        filtered_file_name = filtered_wikidata_path.split('.')[0]+'_'+str(lang)+'.csv'
        groupby_file_name = groupby_entity_dir.split('.')[0]+'_'+str(lang)+'.csv'
        groupby_file_path = os.path.join(groupby_entity_dir,groupby_file_name)
        groupby_text(filtered_file_name,groupby_file_path,lang)
    print("all groupby data saved")

 
#--------agruments (wiki_url path, filtered_wikidata_path)-------
parser = argparse.ArgumentParser()
parser.add_argument("url", help="wiki_url path", type=str)
parser.add_argument("groupby_entity_wikidata", help="groupby_entity_savepath_csv", type=str)

args = parser.parse_args()
url_path=args.url
groupby_entity_savepath=args.groupby_entity_wikidata

#-----download the wiki json file--------
print("----downloading started from url----")
#response = request.urlretrieve(url_path, "latest-all.json.bz2")  #wiki_entity_file_path)
#with urlopen(url_path) as in_stream, open('latest-all.json.bz2', 'wb') as out_file:
#    copyfileobj(in_stream, out_file)
print("----downloading completed----")

#----- Extract the relevant properties of entities from the wiki data ----------------
print("----started extracting relevant entities from the wikidata----")
#getwikiEntities(r"latest-all.json.bz2", r"wikientities_all.out") #commented by ysma, if you run from scratch, open this line
print("----wikidata dump completed----")

#-----filter the wiki data ----------------
print("----started filtering the wikidata----")
get_filtered_entities(filtered_wikidata_path="filtered_wikidata_all.csv", wikioutfile="wikientities_all.out")
print("----filtering done successfully and stored----")

data_save_diff_file(langs,filtered_wikidata_path="filtered_wikidata_all.csv")

#-----group by the entity from filtered wiki data ----------------
print("----started grouping the filtered wikidata----")
#groupby_entity( groupby_entity_dir,filtered_wikidata_path="filtered_wikidata_all.csv")
groupby_entity( groupby_entity_dir=groupby_entity_savepath,filtered_wikidata_path="filtered_wikidata_all.csv")
print("----groupby done successfully and stored in csv file----")







        
 







        
 





        
 





        
 


