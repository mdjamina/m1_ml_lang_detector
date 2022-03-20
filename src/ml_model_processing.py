import pandas as pd
from joblib import Memory, dump

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn import metrics

#https://docs.python.org/fr/3/howto/logging.html
import logging

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
    data_dir = './data/'

    # pickle dataset path file
    corpus_data_path = data_dir + 'data.pkl'

    # pickle output model path file
    output_model_path = data_dir + 'lang_detector.joblib'

    print("\nLoad pickled dataset from file...")
    print(corpus_data_path)
    #Load pickled dataset from file
    data = pd.read_pickle(corpus_data_path) 
    print(f'data size: {data.shape}')

    print("\nSplit datset into random train and test subsets...")

    #la taille de dataset test (33%)
    x_train, x_test, y_train, y_test = train_test_split(data['content'], data['lang'], test_size=0.33,train_size=0.5,  random_state=42) 
    print(f'x_train size (50%):\n {y_train.value_counts()}')

    

    print("\nLinear Support Vector Classification...")
    model = Pipeline([('TfidfVectorizer', TfidfVectorizer(tokenizer=char_tokenizer, ngram_range=(1,2))),
                     ('LinearSVC', LinearSVC()),],verbose=True,memory=mem)
    
    print("\nFit the model...")
    model.fit(x_train, y_train)

    #sauvgarde du modèle 
    print("\nPersist model into file...")
    dump(model, output_model_path) 

    print(f"File: {output_model_path}")


    #calcul des prédictions
    print('\npredict...')
    predictions = model.predict(x_test)

    #df_metrics = 
    
    
    print(metrics.classification_report(y_test,predictions, output_dict=False))

    


if __name__ == "__main__":
    main()
    


    
    
    
