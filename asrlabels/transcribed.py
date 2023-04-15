'''
!pip3 install youtube_dl 
!pip install git+https://github.com/openai/whisper.git
!pip3 install setuptools-rust
!pip3 install ffmpeg-python'''

from __future__ import unicode_literals
import yt_dlp as youtube_dl
import whisper
import argparse
import warnings
warnings.filterwarnings("ignore")
from langid.langid import LanguageIdentifier 
from langid.langid import model as langidmodel
import spacy
import pandas as pd
import os
import sys
import shutil
from multi_rake import Rake


#------filter from wiki_dump file----------------
def save_to_mp3(url, videoFolder=''):
    """Save a YouTube video URL to mp3.

    Args:
        url (str): A YouTube video URL.

    Returns:
        str: The filename of the mp3 file.
    """
    video_id = str(os.path.basename(url)).split('=')[-1]
    videoName = str(os.path.basename(url)).split('=')[-1]+'.mp3'
    videoFile = os.path.join(videoFolder, videoName)
    print(videoFile)

    if os.path.exists(videoFile):
        print("video file already present")
        return videoFile
    videoFile = os.path.join(videoFolder, video_id)
    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl':videoFile,
        'duration':"00:01:00.00",
        'start_time':"00:00:05.00"
    }

    with youtube_dl.YoutubeDL(options) as downloader:
        downloader.download(["" + url + ""])
                
    return downloader.prepare_filename(downloader.extract_info(url, download=False)).replace(".m4a", ".mp3")

 
#--------agruments (youtube url path, asr out path)-------
parser = argparse.ArgumentParser()
parser.add_argument("youtube_url", help="input_video_url", type=str)
parser.add_argument("asroutfile", help="savefile for extracted tokens from sound", type=str)
parser.add_argument("homepath", help="homepath", type=str)

args = parser.parse_args()
youtube_url=args.youtube_url
asroutfile=args.asroutfile
homepath=args.homepath


filename = save_to_mp3(youtube_url, homepath)

filename = filename.split("/")[-1]
filename = filename.split(".")[0]   
trans_filename = homepath+filename+'.mp3'
#filename = 'Martin Luther King, Jr. I Have A Dream Speech-3vDWWy4CMhE.mp3'

try:
    print(homepath+asroutfile)
    df = pd.read_csv(homepath+asroutfile)
    print("********")
    print(homepath+asroutfile)
    if df['filename'].eq(filename).sum() > 0:
        print("Already transcribed before, exit")
        #sys.exit()

except FileNotFoundError:
    print("no records, continue the transcription")


whispermodel = whisper.load_model("base")
print("Just before transcribe")
print(filename)
video_transcription = whispermodel.transcribe(trans_filename, fp16=False)

transcribed_text = video_transcription['text']

print(transcribed_text)

identifier = LanguageIdentifier.from_modelstring(langidmodel, norm_probs=True)

(lang,proba) = identifier.classify(transcribed_text)

langModelMap = dict({'es': "es_core_news_sm",'en': "en_core_web_sm", 'zh':"zh_core_web_sm", })

print(langModelMap.get(lang))

nlp = spacy.load(langModelMap.get(lang))

doc = nlp(transcribed_text)


required_entities = ['CARDINAL','ORDINAL','DATE','TIME']


extracted_tokens = []
for ent in doc.ents:
    if ent.label_ not in required_entities:
        extracted_tokens.append((filename,lang,ent.text, ent.label_),)
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

if not extracted_tokens:
    for token in doc:
        if token.pos_ in ['NOUN','PROPN']:
            print("inside nouns")
            extracted_tokens.append((filename,lang,token.text,token.pos_))

rake = Rake()
keywords = rake.apply(transcribed_text)

rake_keywords = [x[0] for x in keywords][:20]

for token in rake_keywords:
    extracted_tokens.append((filename,lang,token, 'rake'),)

try:
    df = pd.read_csv(homepath+asroutfile)
    increment = pd.DataFrame(columns=['filename','lang','asr_token','token_type'], data = extracted_tokens)
    df = pd.concat([df,increment], ignore_index = True)
except FileNotFoundError:
    df = pd.DataFrame(columns=['filename','lang','asr_token','token_type'], data = extracted_tokens)

df[['filename','lang','asr_token','token_type']].to_csv(homepath+asroutfile)



text_file = open(homepath+filename+".txt", "w")
n = text_file.write(transcribed_text)
text_file.close()
df = df[df.filename == filename]
df[['filename','asr_token','token_type']].to_csv(homepath+filename+".csv", header=False, index=False)











        
 







        
 





        
 





        
 


