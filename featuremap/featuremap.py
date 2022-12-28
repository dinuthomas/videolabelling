
import sys
import json
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
import os

path = sys.argv[1]
id = sys.argv[2].split('.')[0]
dst = sys.argv[3]
sequence_example = open(path, 'rb').read()
x = tf.train.SequenceExample.FromString(sequence_example)
dict_obj = MessageToDict(x)
feat_dict = {}
feat_dict['rgb'] = dict_obj['featureLists']['featureList']["RGB/feature/floats"]['feature']
feat_dict['audio'] = dict_obj['featureLists']['featureList']["AUDIO/feature/floats"]['feature']
dict_json = json.dumps(feat_dict)

out_feat_file = os.path.join(str(dst), str(id) + ".json")

with open(out_feat_file,'w+') as file:
    file.write(dict_json)
    
print("Feats for file %s stored at : %s"%(sys.argv[2], out_feat_file))
