import pandas as pd
from joblib import Memory
from shutil import rmtree

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn import metrics


def char_tokenizer(text):
  """Character Tokenization splits apiece of text 
  into a set of characters.
  """

  return [c for c in text if c not in ' \t\n']


def main():

    # data directory
    data_dir = './data/'

    # corpus
    corpus_data_path = data_dir + 'data.pkl'



    # Create a temporary folder to store the transformers of the pipeline
    mem = Memory(location="/tmp/cachedir", verbose=0)

 
    langs = {'eng': 'English', 'pol': 'Polish', 'deu': 'German', 'fra': 'French', 'spa': 'Spanish', 'ita': 'Italian', 'tur': 'Turkish', 
             'por': 'Portuguese', 'rus': 'Russian', 'ukr': 'Ukrainian', 'nld': 'Dutch', 'bul': 'Bulgarian', 'ell': 'Greek', 'swe': 'Swedish', 
             'hun': 'Hungarian', 'gle': 'Irish', 'lav': 'Latvian', 'dan': 'Danish', 'fin': 'Finnish',
             'ara': 'Arabic', 'heb': 'Hebrew', 'zho': 'Chinese', 'hin': 'Hindi', 'jpn': 'Japanese', 'fas': 'Persian', 'kor': 'Korean',
             'hye': 'Armenian', 'swa': 'Swahili', 'ber': 'Berber', 'kab': 'Kabyle', 'ces': 'Czech', 'lat': 'Latin',
             'nor': 'Norwegian', 'ron': 'Moldavian, Moldovan, Romanian', 'slk': 'Slovak', 'hbs': 'Serbo-Croatian', 'mkd': 'Macedonian',
             'vie': 'Vietnamese', 'est': 'Estonian', 'tha': 'Thai'}




    data = pd.read_pickle(corpus_data_path) 

    test_size= 0.1
    train_size=0.2

    x_train, x_test, y_train, y_test = train_test_split(data['content'], data['lang'], test_size=test_size, train_size=train_size, random_state=1) 


    model = Pipeline([('TfidfVectorizer', TfidfVectorizer(tokenizer=char_tokenizer, ngram_range=(1,2),)),
                     ('LinearSVC', LinearSVC()),],verbose=True,memory=mem)


    model.fit(x_train, y_train)  


    predictions = model.predict(x_test)


    
    df_metrics = pd.DataFrame( metrics.classification_report(y_test,predictions, output_dict=True))


    df_metrics.to_pickle(data_dir +'data_metrics.pkl')




if __name__ == "__main__":
    main()
    


    
    
    
