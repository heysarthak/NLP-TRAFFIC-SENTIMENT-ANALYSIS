import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tweepy as tw
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import spacy 
from spacy import displacy
import json
import random
from spacy.training.example import Example




with open('training_data.json') as fp:
  training_data = json.load(fp)
nlp = spacy.load('en_core_web_sm')

location_data=list()

data=pd.read_csv("tweets.csv")

def clean_text(df):
    ner=nlp.get_pipe("ner")
    losses={}
    for label in training_data["classes"]:
        ner.add_label(label)    
    optimizer = nlp.resume_training()
    for i in range(20):
        random.shuffle(training_data["annotations"])
        for text, annotations in training_data["annotations"]:
            doc1 = nlp.make_doc(text)
            example = Example.from_dict(doc1, annotations)
            nlp.update([example], drop=0.3,losses=losses,sgd=optimizer)   
                
                
    all_reviews = list()
    lines = df["content"].values.tolist()
    for text in lines:
        text = text.lower()
        text=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        text=' '.join(re.sub("(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        pattern = re.compile(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        text = pattern.sub("", text)
        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)   
        
        doc = nlp(text)

        entities = []
        labels = []
        for ent in doc.ents:
            entities.append(ent)
            labels.append(ent.label_)
        df = pd.DataFrame({'Entities':entities,'Labels':labels})
        word_entity=''
        for i in range(len(entities)):
            if (labels[i]=='GPE' or labels[i]=='LOCATION'):
                word_entity=word_entity+' '+str(entities[i])
        location_data.append(word_entity)
        
        #preprocessing 
        tokens = word_tokenize(text)
        table = str.maketrans("", "", string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        PS = PorterStemmer()
        words = [w for w in words if not w in stop_words]
        lemmatizer = nltk.stem.WordNetLemmatizer()
        words= [(lemmatizer.lemmatize(w)) for w in words]
        words = " ".join(words)
        all_reviews.append(words)
    return all_reviews

data['cleanData']=pd.DataFrame(clean_text(data))
data['location_extracted']=pd.DataFrame(location_data)

data['search_word']=data.search_word.map({"accident":0,"car accident":0,"trafficchd":1,"parking space":1,"pot holes":2,"potholes":2,"roadsafetyawareness":1,"road safety awareness":1,"ssptfcchd":1,"crash":0,"personalinjury":0,"traffic":1,"roadjam":1,"road jam":1,"traffic jam":1,"trafficjam":1,"trafficlights":1,"traffic lights":1,"roadblock":1,"road block":1,"chandigarh traffic":1,"chandigarhgtraffic":1,"chandigarh accidents":0,"chandigarhaccidents":0,"chandigarhtrafficlife":1,"rashdriving":1,"baddriver":1})


data.to_csv('FINAL_DATA.csv', index=False)

