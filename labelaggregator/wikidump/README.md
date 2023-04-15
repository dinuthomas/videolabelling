# Setting up the environment 
1. Setup virtual environment and activate it 
$ python3 -m venv </path/to/new/virtual/environment>
$ source </path/to/new/virtual/environment>/bin/activate

2. Install the dependencies using requirements file (in the codebase)
$ pip install -r requirements.txt

3. Download spacy models
$ python3 -m spacy download en_core_web_sm
$ python3 -m spacy download zh_core_web_sm
$ python3 -m spacy download es_core_news_sm


prepare wikidump data
         This contains the folloing steps
           1. download the wikijson data from url (https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2)
           2. Extract the relevant properties of entities from the wiki data and store in an intermediate .out file 
           3. filter the stored wikidata (non relevant entities) and save in a .csv file
           4. finally for the different languages generate corresponding groupby .csv files and stored in a given groupby directory

         Run this scripts:
            $ python wikidump.py <wiki_url>  <out_dir_name>
           
         Example:
            $ python wikidump.py https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2  dump

nohup python wikidump.py https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2  dump &>wikidump.out
           

