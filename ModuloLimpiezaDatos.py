from google.oauth2 import service_account
from google.cloud import bigquery as bq
import pandas as pd
import re
from googletrans import Translator
from textblob import TextBlob
import pycountry
from nltk.corpus import stopwords
from CONTRACTION_MAP import CONTRACTION_MAP
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
from html.parser import HTMLParser
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')
from textblob import TextBlob
from Lematization import  process_text

from LocationDictionary import city_dict
from LocationDictionary import country_dict
from LocationDictionary import ikea_stores_dict

#from pattern import sentiment, mood, modality
#from normalization import normalize_accented_characters, html_parser, strip_html



license_file_path = "C:/Users/diego/Downloads/ingka-food-analytics-dev-2127cf9baa38.json"
credentials = service_account.Credentials.from_service_account_file(license_file_path)
bq_client = bq.Client(credentials=credentials, project=credentials.project_id)

#Descargamos la informacion de la tabla de GCP:
query_string = '''
SELECT * FROM `ingka-food-analytics-dev.TFM_DEV_ingka_food_analytics_dev.Twitter_Feed_Clean_2` where RN=1 ;'''

df = bq_client.query(query_string).to_dataframe()

#Vamos a extraer del texo el usuario que responde el tweet:

df['Inreplay_to'] = df['Text'].apply(lambda x: re.findall(r'@(\w+)', x))

# create a dictionary of city names
city_dict = city_dict
country_dict=country_dict

# function to extract the city name from a list of @mentions
def extract_city(lst):
    for mention in lst:
        # match the @mention against the city names in the dictionary
        for city, aliases in city_dict.items():
            if mention.lower() in aliases:
                return city
    return None

# function to extract the country name from a list of @mentions
def extract_country(lst):
    for mention in lst:
        # match the @mention against the city names in the dictionary
        for city, aliases in country_dict.items():
            if mention.lower() in aliases:
                return city
    return None


# apply the function to the Inreplay_to column and store the result in a new column
df['TextReplayLocationCity'] = df['Text'].apply(lambda x: extract_city(x))
df['TextReplayLocationCountry'] = df['Text'].apply(lambda x: extract_country(x))

#Se crea un diccionario para identificar los tweets de las cuentas de IKEA: 

id_to_name = ikea_stores_dict

df['Is_Ikea'] = df['Author_id'].map(id_to_name)



## 1_Creamos una funcion para quitar los elementos de URL y links:
def remove_usernames_links(tweet):
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('http[^\s]+','',tweet)
    return tweet
df['Text_NotURL'] = df['Text'].apply(remove_usernames_links)

# Initialize translator
translator = Translator()

# Define function to translate text to English

def translate_to_english(text, lang):
    try:
        # If the text is already in English, return it
        if lang == 'en':
            return text
        # Otherwise, translate the text to English
        else:
            translation = translator.translate(text, dest='en', src=lang)
            return translation.text
    except ValueError as e:
        # If the language is not recognized, return the original text
        return text
    except Exception as e:
        # If there's any other error, return None so that it can be handled later
        return None
df['English_Text'] = df.apply(lambda row: translate_to_english(row['Text_NotURL'], row['Language']), axis=1)



#2_Se crea una funcion para quitar las contractiones y expandir la frase:Ejemplo:Don't por do not.

def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction 
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence

df['expanded_corpus']= [expand_contractions(sentence, CONTRACTION_MAP)
   for sentence in df['English_Text']]

##_3_Empezamos por crear una funcion que quita caracteres
def remove_characters_before_tokenization(sentence,keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
       PATTERN = r'[?|$|&|*|%|@|(|)|~]' 
       filtered_sentence = re.sub(PATTERN, r'', sentence)
    else:
       PATTERN = r'[^a-zA-Z0-9 ]' # only extract alpha-numeric characters
       filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence

def remove_stopwords(text, excluded_words=[]):
    tokens = text.split()
    stopwords_set = set(stopwords.words('english')) - set(excluded_words)
    return [t for t in tokens if t.lower() not in stopwords_set]

excluded_words = ['to', 'the', 'and', 'food', 'our', 'more', 'a', 'of', 'in','ikea','food','with','we'
                'is','by','can','at','for','by','were','we','is','on','your','you','make','amp','that',
                'this','be','as','from','here','will','are','how','about','only','not','do' ]


df['filtered_list_2'] = [remove_characters_before_tokenization(sentence)
  for sentence in df['expanded_corpus']]


df['LematizedText'] = df['filtered_list_2'].apply(process_text)


##_4_Vamos ahora a dejar todo en minusculas:

df['LowerText'] = df['LematizedText'].str.lower()

df['StopWordsText'] = df['LowerText'].apply(remove_stopwords)


##Vamos a generar un campo nuevo que nos indique la cantidad de veces que ese texto esta en nuestra base
df['RN'] = df.sort_values(['id','id'], ascending=[True,False]) \
             .groupby(['filtered_list_3']) \
             .cumcount() + 1


print(df)
#df.to_excel('TFM_Clean_ModeloESPAÑOL_Final.xlsx',engine='xlsxwriter')
#Creo los campos para la nueva tabla en BQ con la nueva informacion

job_config = bq.LoadJobConfig(
schema=[
    bq.SchemaField('Author_id', 'STRING'),
    bq.SchemaField('created_at_short', 'Date'),
    bq.SchemaField('Retweet_Count', 'INTEGER'),
    bq.SchemaField('Like_Count', 'INTEGER'),
    bq.SchemaField('Replie_Count', 'INTEGER'),
    bq.SchemaField('id', 'STRING'),
    bq.SchemaField('Language', 'STRING'),
    bq.SchemaField('Text', 'STRING'),
    bq.SchemaField('Keyword', 'STRING'),
    bq.SchemaField('User_Location', 'STRING'),
    bq.SchemaField('Is_Retweet', 'INTEGER'),
    bq.SchemaField('Is_Replied', 'INTEGER'),
    bq.SchemaField('TextReplayLocationCity', 'INTEGER'),
    bq.SchemaField('TextReplayLocationCountry', 'INTEGER'),
    bq.SchemaField('Is_Publication', 'INTEGER'),
    bq.SchemaField('RN', 'INTEGER'),
    bq.SchemaField('Is_Ikea', 'STRING')	,
	bq.SchemaField('Text_NotURL', 'STRING'),	
    bq.SchemaField('expanded_corpus', 'STRING')	,
    bq.SchemaField('filtered_list_2', 'STRING')	,
    bq.SchemaField('filtered_list_3', 'STRING')	
    ]
)

table_id = 'ingka-food-analytics-dev.TFM_DEV_ingka_food_analytics_dev.Twitter_Feed_Model_Final'  #Nombre de la tabla
load_job = bq_client.load_table_from_dataframe(df, table_id, job_config=job_config) #Cambiar df_all por el nombre del dataframe que queráis cargar
load_job.result()

