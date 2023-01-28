'''
!pip3 install youtube_dl 
!pip install git+https://github.com/openai/whisper.git
!pip3 install setuptools-rust
!pip3 install ffmpeg-python'''

from __future__ import unicode_literals
import youtube_dl
import whisper
import argparse
import warnings
warnings.filterwarnings("ignore")
from langid.langid import LanguageIdentifier 
from langid.langid import model as langidmodel
#import spacy
from multi_rake import Rake
import pandas as pd
import os


#------filter from wiki_dump file----------------
def save_to_mp3(url, videoFolder=''):
    """Save a YouTube video URL to mp3.

    Args:
        url (str): A YouTube video URL.

    Returns:
        str: The filename of the mp3 file.
    """

    videoName = str(os.path.basename(url)).split('=')[-1]+'.mp3'
    videoFile = os.path.join(videoFolder, videoName)

    if os.path.exists(videoFile):
        print("video file already present")
        return videoFile

    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl':videoFile,
    }

    with youtube_dl.YoutubeDL(options) as downloader:
        downloader.download(["" + url + ""])
                
    return downloader.prepare_filename(downloader.extract_info(url, download=False)).replace(".m4a", ".mp3")

 
#--------agruments (youtube url path, asr out path)-------
parser = argparse.ArgumentParser()
parser.add_argument("youtube_url", help="input_video_url", type=str)
parser.add_argument("asroutfile", help="savefile for extracted tokens from sound", type=str)

args = parser.parse_args()
youtube_url=args.youtube_url
asroutfile=args.asroutfile


filename = save_to_mp3(youtube_url)
#filename = 'Martin Luther King, Jr. I Have A Dream Speech-3vDWWy4CMhE.mp3'

whispermodel = whisper.load_model("base")

video_transcription = whispermodel.transcribe(filename, fp16=False)

print(video_transcription['text'])

identifier = LanguageIdentifier.from_modelstring(langidmodel, norm_probs=True)

(lang,proba) = identifier.classify(video_transcription['text'])

langModelMap = dict({'es': "es_core_news_sm",'en': "en_core_web_sm", 'zh':"zh_core_web_sm", })

print(langModelMap.get(lang))

rake = Rake()
keywords = rake.apply(video_transcription['text'])


required_entities = ['CARDINAL','ORDINAL','DATE','TIME']


extracted_tokens = []
rake_keywords = [x[0] for x in keywords][:20]

for token in rake_keywords:
    extracted_tokens.append((filename,lang,token, 'rake'),)

try:
    df = pd.read_csv(asroutfile)
    increment = pd.DataFrame(columns=['filename','lang','asr_token','token_type'], data = extracted_tokens)
    df = pd.concat([df,increment], ignore_index = True)
except FileNotFoundError:
    df = pd.DataFrame(columns=['filename','lang','asr_token','token_type'], data = extracted_tokens)

df[['filename','lang','asr_token','token_type']].to_csv(asroutfile)











        
 







        
 





        
 





        
 


