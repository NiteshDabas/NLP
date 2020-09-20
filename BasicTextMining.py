#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nitesh Dabas
"""

#import numpy and pandas
import numpy as np
import pandas as pd

#load the dataset
train = pd.read_csv('/Users/dabasn/Desktop/Nitesh/Studies/Projects/NLP/BasicTextMining/BasicTextMining.csv')
train.head()

#Count number of words per document/tweet
train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))
train[['tweet','word_count']]

#Count number of characters per document/tweet
train['char_count'] = train['tweet'].str.len() ## this also includes spaces
train[['tweet','char_count']].head()

#Function to calculate Average Word Length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

#calling function for Average Word Length calculation for  eac tweet
train['avg_word'] = train['tweet'].apply(lambda x: avg_word(x))
train[['tweet','avg_word']].head()

#Number of stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['tweet','stopwords']].head()

#Number of special characters
train['hastags'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['tweet','hastags']].head()

#Number of numerics
train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['tweet','numerics']].head()

#Number of Uppercase words
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()

#Number of Uppercase words
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()

#Removing Punctuation
train['tweet'] = train['tweet'].str.replace('[^\w\s]','')
train['tweet'].head()

#Removal of Stop Words
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['tweet'].head()

#Common word removal
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

#Rare words removal
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()

#Spelling correction
from textblob import TextBlob
train['tweet'][:10].apply(lambda x: str(TextBlob(x).correct()))

# Tokenization
WordList=TextBlob(train['tweet'][1]).words
WordList

#Stemming
pd.set_option('display.max_colwidth', -1)
from nltk.stem import PorterStemmer
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#Lemmatization
from textblob import Word
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['tweet'].head()

#Bigram
WordList = TextBlob(train['tweet'][0]).ngrams(2)
WordList

#Term frequency
tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1

#Inverse Document Frequency
for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))
tf1

#Term Frequency – Inverse Document Frequency (TF-IDF)
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1

#Sentiment Analysis
train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
train[['tweet','sentiment']].head()