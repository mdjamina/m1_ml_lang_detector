import pandas as pd

def main():

    # data directory
    data_dir = './data/'

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
    data = pd.read_csv(corpus_file_path, sep='\t',
                       header=None, compression='bz2')
    data.columns = ['id', 'I_Id', 'content']

    # suppression des valeurs NAN
    data.dropna(subset=["I_Id"], inplace=True)

    # chargement du fichier macrolanguages
    macro_lang = pd.read_csv(macrolanguages_path, sep='\t')
    macro_lang.columns = ['lang', 'I_Id',	'I_Status']

    # jointure entre la table des sentences et la table macrolanguages
    data = pd.merge(data, macro_lang, how='left', on='I_Id')
    data.lang.fillna(data.I_Id, inplace=True)

    # suppression des columns 'I_Status', 'id' et 'I_Id'
    data = data.drop(columns=['I_Status', 'id', 'I_Id'])

    # garder que les langues selectionnées dans la liste langs
    data = data[data['lang'].isin(langs.keys())]


    data.to_pickle(data_dir +'data.pkl')




if __name__ == "__main__":
    main()
    


    
    
    
