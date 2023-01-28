import argparse
import csv
import pandas as pd
import pickle
import os
import ast

lang_dict = {'en':'english', 'es':'spanish', 'zh':'chinese','ar':'arabic'}
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="input_file path", type=str)
parser.add_argument("automaton_saved_dir", help="automaton_saved_dir", type=str)
parser.add_argument("output_file", help="output_file path", type=str)
args = parser.parse_args()
automaton_saved_path = args.automaton_saved_dir
input_file_path=args.input_file
output_file_path=args.output_file

#---reading input csv file--------------   
df=pd.read_csv(input_file_path,index_col=False, encoding = "ISO-8859-1", engine ='python').fillna("")
print("---input csv file reading done---")

print(df.head())

#---load automaton pickle files-----
print("----loading pickle files----")
listOf_dir = os.listdir(automaton_saved_path)
atm = {}

for file_name in listOf_dir:
    lang = 'en'
    file_path = os.path.join(automaton_saved_path)
    atm_lang = []
    lang = str(file_name).split('_')[1]
    file_path = os.path.join(automaton_saved_path,file_name)
    atm_lang = []
    listOf_atm = os.listdir(file_path)

    for atm_name in listOf_atm:
        if atm_name.endswith('.pickle'):
            pickle_file = os.path.join(file_path,atm_name)
            print(pickle_file)
            with open(pickle_file,'rb') as f:
                A = pickle.load(f)
                atm_lang.append(A)
    print("done")
    atm[lang]=atm_lang

print("------{} pickle files loaded succsessfully-----".format(len(atm)))

#---find keywords using automaton----
def find_keywords(line, At):
    found_keywords = []
    for end_index, (cat, keyw) in At.iter(line):
        val = (cat, keyw)
        print("---oh my god, found it----")
        found_keywords.append(val)
    return found_keywords
    

ids = df['filename'].unique()

outdf = None
columns=['vid','taglist','wikiid']
token_id_comb = []
for id in ids:
    final_result = []
    print(id)
    item_df = df[df.filename == id]
    #item_df['asr_token'] = item_df.asr_token.map(lambda x: str(x).replace(' ',''))
    taglist = list(item_df.asr_token)
    vid_doc = (' '+' '.join(taglist)+' ').lower()

    lang = ''.join(item_df.lang.unique())
    print(''.join(item_df.lang.unique()))

    for at in atm[lang]:
        result = find_keywords(vid_doc, at)
        final_result += result

    print(final_result)
    token_id_comb.append((id,taglist,final_result))
    print(token_id_comb)
    try:
        outdf = pd.read_csv(output_file_path)
        increment = pd.DataFrame(columns=columns, data = token_id_comb)
        outdf = pd.concat([outdf,increment], ignore_index = True)
    except FileNotFoundError:

        outdf = pd.DataFrame(columns=columns, data=token_id_comb)


outdf[columns].to_csv(output_file_path)
print("\n---stored output csv file successfully---")

