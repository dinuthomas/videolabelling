import pandas as pd
import os

df = pd.read_excel('./data/VideoTestList_1.xlsx', sheet_name = 'updated')
df.columns = ['category', 'video_url']

vocab = pd.read_csv('./vocabulary_youtube_8m.csv')
ids = []
for _, row in df.iterrows():
    url = row['video_url']
    id = url.split('/')[-1]
    ids.append(id)
df['id'] = ids

for dirpath, dirnames, filenames in os.walk('./data/pred_yt8m'):
    for filename in filenames:
        path = os.path.join(dirpath, filename)
        id = filename.split('/')[-1].split('.')[0]
        
        pred = pd.read_csv(path)
        pred.fillna('', inplace=True)
        pred['value'] = pred['value'].apply(str)
        #print(pred.iloc[:21, 2])
        result = [vocab[vocab['Name']==i]['Vertical1'].values[0] if i != '' else '' for i in pred.iloc[:21, 2]]
        name = ','.join(pred.iloc[:21, 2])
        value = ' '.join(pred.iloc[:21, 3])
        df.loc[df['id']==id, 'name'] = name
        df.loc[df['id']==id, 'value'] = value
        df.loc[df['id']==id, 'vertical1'] = ','.join(result)

df.to_csv('results.csv')