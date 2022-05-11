# Insert your code here.
from joblib import load

langs = {'eng': 'English', 'pol': 'Polish', 'deu': 'German', 'fra': 'French', 'spa': 'Spanish', 'ita': 'Italian', 'tur': 'Turkish',
         'por': 'Portuguese', 'rus': 'Russian', 'ukr': 'Ukrainian', 'nld': 'Dutch', 'bul': 'Bulgarian', 'ell': 'Greek', 'swe': 'Swedish',
         'hun': 'Hungarian', 'gle': 'Irish', 'lav': 'Latvian', 'dan': 'Danish', 'fin': 'Finnish',
         'ara': 'Arabic', 'heb': 'Hebrew', 'zho': 'Chinese', 'hin': 'Hindi', 'jpn': 'Japanese', 'fas': 'Persian', 'kor': 'Korean',
         'hye': 'Armenian', 'swa': 'Swahili', 'ber': 'Berber', 'kab': 'Kabyle', 'ces': 'Czech', 'lat': 'Latin',
         'nor': 'Norwegian', 'ron': 'Moldavian, Moldovan, Romanian', 'slk': 'Slovak', 'hbs': 'Serbo-Croatian', 'mkd': 'Macedonian',
         'vie': 'Vietnamese', 'est': 'Estonian', 'tha': 'Thai'}


def char_tokenizer(text):
    """Character Tokenization splits apiece of text 
    into a set of characters.
    """
    return [c for c in text if c not in ' \t\n']


model = load('../data/lang_detector.joblib')


class Languge:

    def __init__(self) -> None:
        self.model = model

    def detect(self, text):

        lang = self.model.predict([text])[0]

        lib = langs[lang]

        return (lang,lib)


language = Languge()

if __name__ == '__main__':


    text = """Этот модуль предоставляет имена для многих типов, необходимых для реализации интерпретатора Python. 
   Он намеренно избегает включения некоторых типов, возникающих лишь случайно во время обработки, таких как тип listiterator."""

    print(language.detect(text))  


    text ='توفر هذه الوحدة أسماء للعديد من الأنواع المطلوبة لتنفيذ مترجم بايثون\
      . يتجنب عمدًا تضمين بعض الأنواع التي تظهر بشكل عرضي فقط أثناء المعالجة مثل نوع قائمة القوائم.'

    print(language.detect(text))  

    text = 'מודול זה מספק שמות לרבים מהסוגים הנדרשים ליישום מתורגמן Python.\
       הוא נמנע בכוונה מלהכליל כמה מהסוגים שעולים רק במקרה במהלך העיבוד, כגון סוג הליסטיטרטור.'
    print(language.detect(text))  


    text = 'Bu modül, bir Python yorumlayıcısını uygulamak için gereken birçok tür için adlar sağlar.\
       Listeleyici türü gibi, işleme sırasında yalnızca tesadüfen ortaya çıkan bazı türleri dahil etmekten kasıtlı olarak kaçınır.'
    print(language.detect(text))  


  




  