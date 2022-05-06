import pandas as pd


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
    

def main():

    # data directory
    data_dir = '../data/'

    # corpus
    corpus_file_path = data_dir + 'sentences.tar.bz2'

    macrolanguages_path = data_dir +'iso-639-3-macrolanguages.tab'

    langs = {'eng': 'English', 'pol': 'Polish', 'deu': 'German', 'fra': 'French', 'spa': 'Spanish', 'ita': 'Italian', 'tur': 'Turkish', 
             'por': 'Portuguese', 'rus': 'Russian', 'ukr': 'Ukrainian', 'nld': 'Dutch', 'bul': 'Bulgarian', 'ell': 'Greek', 'swe': 'Swedish', 
             'hun': 'Hungarian', 'gle': 'Irish', 'lav': 'Latvian', 'dan': 'Danish', 'fin': 'Finnish',
             'ara': 'Arabic', 'heb': 'Hebrew', 'zho': 'Chinese', 'hin': 'Hindi', 'jpn': 'Japanese', 'fas': 'Persian', 'kor': 'Korean',
             'hye': 'Armenian', 'swa': 'Swahili', 'ber': 'Berber', 'kab': 'Kabyle', 'ces': 'Czech', 'lat': 'Latin',
             'nor': 'Norwegian', 'ron': 'Moldavian, Moldovan, Romanian', 'slk': 'Slovak', 'hbs': 'Serbo-Croatian', 'mkd': 'Macedonian',
             'vie': 'Vietnamese', 'est': 'Estonian', 'tha': 'Thai'}

    # Chargement du corpus
    logger.info(f"Load corpus from file {corpus_file_path}...")
    data = pd.read_csv(corpus_file_path, sep='\t',
                       header=None, compression='bz2')
    data.columns = ['id', 'I_Id', 'content']

    # suppression des valeurs NAN
    data.dropna(subset=["I_Id"], inplace=True)

    # chargement du fichier macrolanguages
    logger.info(f"Load macrolanguages file {macrolanguages_path}...")
    macro_lang = pd.read_csv(macrolanguages_path, sep='\t')
    macro_lang.columns = ['lang', 'I_Id',	'I_Status']

    logger.info(f"Preprocessing start...")

    # jointure entre la table des sentences et la table macrolanguages
    data = pd.merge(data, macro_lang, how='left', on='I_Id')
    data.lang.fillna(data.I_Id, inplace=True)

    # suppression des columns 'I_Status', 'id' et 'I_Id'
    data = data.drop(columns=['I_Status', 'id', 'I_Id'])

    # garder que les langues selectionnées dans la liste langs
    data = data[data['lang'].isin(langs.keys())]

    logger.info(f"save result in '{data_dir}data.pkl'...")


    data.to_pickle(data_dir +'data1.pkl')


    logger.info(f"Preprocessing end...")




if __name__ == "__main__":

    logger = getLogger("data_pre_processing")
    main()
    


    
    
    
