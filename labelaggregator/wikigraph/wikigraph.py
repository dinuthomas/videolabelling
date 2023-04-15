import pickle
import ahocorasick as ahc
import argparse
import os
import pandas as pd
import shutil
import ast

lang_dict = {'en':'english'}

#-----make automaton for entities----
def make_aho_automaton(keywords):
    A = ahc.Automaton()  # initialize
    for i, (key, cat) in enumerate(keywords):
        A.add_word(key, (cat, key)) # add keys and categories%
        if i%500000==0:
            print(i)        
    A.make_automaton() # generate automaton
    return A

def extract_keywords(file_path, start_indx, end_indx):
    df = pd.read_csv(file_path)

    key_words_new = []
    for i in range(start_indx, end_indx):

        entity = df['entity_txt'][i]
        entity = str(entity).replace(" ","#")
        key_new = " "+str(entity)+" "

        cat_new1 = ast.literal_eval(df['entity_ids'][i])
        cat_new2 = ast.literal_eval(df['sitelinks'][i])
        cat_new = dict(zip(cat_new1, cat_new2))
        keys = (key_new, cat_new)
        key_words_new.append(keys)
        #print(keys)
        if i%50000==0:
            print(i) 
            #break
    return key_words_new

def getlength(file_path):
    df = pd.read_csv(file_path)
    length = len(list(df['entity_txt']))
    lang = list(df['entity_lang'])[1]
    return length, lang

    
#----arguments are input groupby dir path, no.s of automaton wants to generate and automaton save dir
parser = argparse.ArgumentParser()
parser.add_argument("groupby_dir", help="groupby_entity_dir_path", type=str)
parser.add_argument("automaton_size", help="give no. of automaton want to generate", type=int)
parser.add_argument("automaton_save_dir", help="give automaton save directory ", type=str)

args = parser.parse_args()
grouped_dir = args.groupby_dir
automaton_size = args.automaton_size
automata_save_dir= args.automaton_save_dir

if not os.path.exists(automata_save_dir):
    os.makedirs(automata_save_dir)
print("load groupby files")

files = os.listdir(grouped_dir)
for file in files:

    #-----------loading the input groupby files
    print("input file loading...")
    file_path = str(grouped_dir)+'/'+str(file)
    print(file_path)
    keywords_length, lang = getlength(file_path)

    automata_dir = str(automata_save_dir)+ '/' +str(automata_save_dir)+'_'+lang
    if os.path.exists(automata_dir):
        shutil.rmtree(automata_dir)

    os.makedirs(automata_dir)
    print(automata_dir)

    key_words_split=[]
    for i in range(automaton_size):    
        start_indx = i*((keywords_length)//automaton_size)
        end_indx = (i+1)*((keywords_length)//automaton_size)
        print(start_indx,end_indx)
        new_keywords = extract_keywords(file_path, start_indx, end_indx)
        At=make_aho_automaton(new_keywords)
        print("automaton{} generated successfully".format(i+1))
        automaton_file_name ="automaton"+str(i+1)+".pickle"
        automaton_file_path = os.path.join(automata_dir,automaton_file_name)
        with open(automaton_file_path, 'wb') as f:
            pickle.dump(At, f)
        print("automaton{} saved successfully".format(i+1))




