
import sys
import json
import pandas as pd
import numpy as np
import os

vocab_path = sys.argv[1]
tag_results = sys.argv[2]
mapped_out = sys.argv[3]
print(vocab_path)
print(tag_results)


results = pd.read_csv(tag_results)

vocab = pd.read_csv(vocab_path)

vocab = vocab[['Index', 'Name']]

model_pred = results.dropna()[['id','idx','value']]

model_pred['label'] = model_pred.idx.map(lambda x:x.split(','))
model_pred['conf'] = model_pred.value.map(lambda x:x.split(','))

preds1 = model_pred[['id','label']].explode('label')
preds2 = model_pred[['id','conf']].explode('conf')

finalpred = pd.concat([preds1,preds2], axis=1)[['id','label','conf']]

finalpred['label']=finalpred.label.astype(int)

mapped_pred = pd.merge(finalpred, vocab, how='left', left_on='label', right_on='Index')

mapped_result = mapped_pred.T.drop_duplicates().T[['id', 'label', 'conf', 'Name']]

mapped_result.drop_duplicates(['id','label'],inplace=True)

mapped_result.to_csv(mapped_out)