#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
from joblib import Memory, dump

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn import metrics

#https://docs.python.org/fr/3/howto/logging.html
import logging

global logger

def getLogger(name):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger
    

# Create a temporary folder to store the transformers of the pipeline
mem = Memory(location="/tmp/cachedir", verbose=0)

def char_tokenizer(text):
  """Character Tokenization splits apiece of text 
  into a set of characters.
  """

  return [c for c in text if c not in ' \t\n']


def get_model(x,y,tokenizer=None,ngram_range=(1,1), verbose=True, memory=None ):
    model = Pipeline([('TfidfVectorizer', TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram_range)),
                     ('LinearSVC', LinearSVC()),],verbose=verbose,memory=memory)
    model.fit(x, y)

    return model
    


def main():



    # data directory
    data_dir = '../data/'

    # pickle dataset path file
    corpus_data_path = data_dir + 'data.pkl'

    # pickle output model path file
    output_model_path = data_dir + 'lang_detector.joblib'

    logger.info(f"Load pickled dataset from file {corpus_data_path}...")
    #Load pickled dataset from file
    data = pd.read_pickle(corpus_data_path) 
    #print(f'data size: {data.shape}')

    logger.info("Split datset into random train and test subsets...")
    #la taille de dataset test (33%)
    x_train, x_test, y_train, y_test = train_test_split(data['content'], data['lang'], test_size=0.33,train_size=0.5,  random_state=42) 
    #print(f'x_train size (50%):\n {y_train.value_counts()}')

    

    logger.info("Set Pipeline, Linear Support Vector Classification...")
    model = Pipeline([('TfidfVectorizer', TfidfVectorizer(tokenizer=char_tokenizer, ngram_range=(1,2))),
                     ('LinearSVC', LinearSVC()),],verbose=True,memory=mem)
    
    logger.info("Fit the model...")
    model.fit(x_train, y_train)

    #sauvgarde du modèle 
    logger.info("Persist model into file...")
    dump(model, output_model_path) 

    logger.info(f"File: {output_model_path}")


    #calcul des prédictions
    logger.info('predict...')
    predictions = model.predict(x_test)

    #df_metrics = 
    
    
    print(metrics.classification_report(y_test,predictions, output_dict=False))

    


if __name__ == "__main__":

    logger = getLogger("ml_model_processing")
    main()
   


    
    
    
