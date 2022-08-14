import pandas as pd
import re
import string

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict

#Import the test and train files
dftrain = pd.read_csv('../Files/train.csv', usecols=["id", "keyword","location", "text", "target"])
dftest = pd.read_csv('../Files/test.csv', usecols=["id", "keyword","location", "text"])

#Adapted from documentation at: https://github.com/cbaziotis/ekphrasis
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def preprocess(txt):
    #Ignore non ascii characters
    output = str(txt).encode('ascii', 'ignore').decode()

    output = re.sub(r'%20', ' ', output)

    #Replace slang
    list = []
    for word in output.split():
        if word.lower() in slangdict.keys():
            list.append(slangdict[word.lower()])
        else:
            list.append(word)
    output = " ".join(list).strip()

    #Expand contractions in the text
    output = re.sub(r"n\'t", " not", output)
    output = re.sub(r"\'re", " are", output)
    output = re.sub(r"\'s", " is", output)
    output = re.sub(r"\'d", " would", output)
    output = re.sub(r"\'ll", " will", output)
    output = re.sub(r"\'t", " not", output)
    output = re.sub(r"\'ve", " have", output)
    output = re.sub(r"\'m", " am", output)
    output = re.sub(r"u s", "united states", output)
    output = re.sub(r"usa", "united states", output)

    #Apply the preprocessor using the example from the documentation above
    output = " ".join(text_processor.pre_process_doc(output))
    output.strip()

    #Remove the tags refering to repeated
    for patt in [r"<repeated>"]:
        output = re.sub(patt, '', output)

    #Replace slang again after segmenting (for example for hashtags)
    list = []
    for word in output.split():
        if word.lower() in slangdict.keys():
            list.append(slangdict[word.lower()])
        else:
            list.append(word)
    output = " ".join(list).strip()

    #Remove punctuation and whitespace
    output = re.sub(r'[%s]' % re.escape(''.join(string.punctuation)), r' ',output)
    output = re.sub(r'\s+', ' ', output)

    return output.strip().lower()

#Create the new dataframes and apply cleaning to the text, location, and keywords
dfcleantrain = pd.DataFrame().reindex_like(dftrain)
dfcleantest = pd.DataFrame().reindex_like(dftest)
dfcleantrain["id"] = dftrain["id"]
dfcleantrain["keyword"] = dftrain["keyword"].apply(preprocess)
dfcleantrain["location"] = dftrain["location"].apply(preprocess)
dfcleantrain["text"] = dftrain["text"].apply(preprocess)
dfcleantrain["target"] = dftrain["target"]
dfcleantest["id"] = dftest["id"]
dfcleantest["keyword"] = dftest["keyword"].apply(preprocess)
dfcleantest["location"] = dftest["location"].apply(preprocess)
dfcleantest["text"] = dftest["text"].apply(preprocess)

dfcleantrain.to_csv("../Files/cleantrain.csv", index=False)
dfcleantest.to_csv("../Files/cleantest.csv", index=False)