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
import spacy
import pandas as pd


#------filter from wiki_dump file----------------
def save_to_mp3(url):
    """Save a YouTube video URL to mp3.

    Args:
        url (str): A YouTube video URL.

    Returns:
        str: The filename of the mp3 file.
    """

    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
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


#filename = save_to_mp3(youtube_url)
filename = 'Martin Luther King, Jr. I Have A Dream Speech-3vDWWy4CMhE.mp3'

whispermodel = whisper.load_model("base")

video_transcription = whispermodel.transcribe(filename, fp16=False)

print(video_transcription['text'])

identifier = LanguageIdentifier.from_modelstring(langidmodel, norm_probs=True)

(lang,proba) = identifier.classify(video_transcription['text'])

langModelMap = dict({'es': "es_core_news_sm",'en': "en_core_web_sm", 'zh':"zh_core_web_sm", })

print(langModelMap.get(lang))

nlp = spacy.load(langModelMap.get(lang))

doc = nlp(video_transcription['text'])


required_entities = ['PERSON', 'GPE', 'NORP']


extracted_tokens = []
for ent in doc.ents:
    if ent.label_ in required_entities:
        extracted_tokens.append((ent.text, ent.label_))
        print(ent.text, ent.start_char, ent.end_char, ent.label_)

df = pd.DataFrame(columns=['asr_token','token_type'], data = extracted_tokens)
df.to_csv(asroutfile)











        
 







        
 





        
 





        
 


