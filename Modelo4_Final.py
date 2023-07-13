import tensorflow_datasets as tfds
import pandas as pd
import re
import numpy as np
from CONTRACTION_MAP import CONTRACTION_MAP
import re
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

stop_words = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from google.oauth2 import service_account
from google.cloud import bigquery as bq
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd


from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#from DataCleaning1_3 import extract_city,extract_country
from LocationDictionary import city_dict
from LocationDictionary import country_dict



ds = tfds.load('sentiment140', split='train')
data = []
for example in ds:
#for example in ds.take(500000):
    text = example['text'].numpy().decode('utf-8')
    polarity = example['polarity'].numpy()
    data.append((text, polarity))

df = pd.DataFrame(data, columns=['text', 'polarity'])

def remove_usernames_links(tweet):
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('http[^\s]+','',tweet)
    if len(tweet.strip()) == 0:
        return ''
    else:
        return tweet
df['Text_NotURL'] = df['text'].apply(remove_usernames_links)


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

df['Text_Expanded']= [expand_contractions(sentence, CONTRACTION_MAP)
   for sentence in df['Text_NotURL']]

def remove_characters_before_tokenization(sentence,keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
       PATTERN = r'[?|$|&|*|%|@|(|)|~]' 
       filtered_sentence = re.sub(PATTERN, r'', sentence)
    else:
       PATTERN = r'[^a-zA-Z0-9 ]' # only extract alpha-numeric characters
       filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence

df['Text_RemoveCharacters'] = [remove_characters_before_tokenization(sentence)
  for sentence in df['Text_Expanded']]

df['Text_Lowercase'] = df['Text_RemoveCharacters'].str.lower()

def process_text(text):
    # Remove repeated characters
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(w) for w in TextBlob(text).words])
    # Correct spelling
    #textblob = TextBlob(text)
    #exclude_words=['ikea','meatballs','meatball','ikeas']
    #for word in textblob.words:
       #if word not in exclude_words:
     #       text = text.replace(word, str(TextBlob(word).correct()))    

    return text

df['Text_Lematized'] = df['Text_Lowercase'].apply(process_text)

def remove_stopwords(text, excluded_words=[]):
    stopwords_set = set(stopwords.words('english')) - set(excluded_words)
    pattern = re.compile(r'\b(' + '|'.join(stopwords_set) + r')\b')
    return re.sub(pattern, '', text)

excluded_words = ['to', 'the', 'and', 'food', 'our', 'more', 'a', 'of', 'in','ikea','food','with','we',
                'is','by','can','at','for','by','were','we','is','on','your','you','make','amp','that',
                'this','be','as','from','here','will','are','how','about','only','not','do','do not' ]

#df['Text_Stopword'] = df['Text_Lematized'].apply(remove_stopwords)



# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df['Text_Lematized'], df['polarity'], test_size=0.3, random_state=42, stratify=df['polarity'])

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer(min_df=5, ngram_range=(1, 6))
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)



# Create a LinearSVC model object and fit it on the training data
model = LinearSVC(random_state=42,                   tol=0.01,
                   C=1,
                   max_iter=10000,
                   penalty='l2',
                   loss='hinge')#, C=100)
                  
calibrated_model = CalibratedClassifierCV(model)

calibrated_model.fit(X_train_tf, Y_train)
train_predictions = calibrated_model.predict(X_train_tf)

train_accuracy = accuracy_score(Y_train, train_predictions)
train_precision = precision_score(Y_train, train_predictions, average='macro')
train_recall = recall_score(Y_train, train_predictions, average='macro')
train_f1 = f1_score(Y_train, train_predictions, average='macro')

# Make predictions on the test set
test_predictions = calibrated_model.predict(X_test_tf)

# Calculate metrics for the test set
test_accuracy = accuracy_score(Y_test, test_predictions)
test_precision = precision_score(Y_test, test_predictions, average='macro')
test_recall = recall_score(Y_test, test_predictions, average='macro')
test_f1 = f1_score(Y_test, test_predictions, average='macro')

# Print the metrics
print("Metrics for the training set:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1-Score:", train_f1)
print()
print("Metrics for the test set:")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1-Score:", test_f1)

train_samples = len(X_train)
print("Number of observations in the training set:", train_samples)

# Number of observations in the test set
test_samples = len(X_test)
print("Number of observations in the test set:", test_samples)
license_file_path = "C:/Users/diego/Downloads/ingka-food-analytics-dev-2127cf9baa38.json"
credentials = service_account.Credentials.from_service_account_file(license_file_path)
bq_client = bq.Client(credentials=credentials, project=credentials.project_id)


query_string = '''
SELECT * FROM  `ingka-food-analytics-dev.TFM_DEV_ingka_food_analytics_dev.Twitter_Feed_Model_FinalDataSet`  where Keyword is not null
and Is_Publication=1 AND RN=1 and RelevantTweet=1  
;'''

df_test = bq_client.query(query_string).to_dataframe()

import spacy

nlp = spacy.load("en_core_web_sm")

import spacy
from spacy.matcher import PhraseMatcher

def extract_city(text):
    # load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # create a PhraseMatcher to match city names
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for city, aliases in city_dict.items():
        patterns = [nlp.make_doc(alias) for alias in aliases]
        matcher.add(city, None, *patterns)
    
    # add a rule to recognize "Barcelona" as a location entity
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "GPE", "pattern": [{"LOWER": "barcelona"}]}]
    ruler.add_patterns(patterns)
    
    # use spaCy to extract entities from text
    doc = nlp(text)
    
    # iterate over the entities and check if they are in the cities dictionary
    for ent in doc.ents:
        if ent.label_ == "GPE":
            for city, aliases in city_dict.items():
                if ent.text.lower() in aliases:
                    return city
    
    # if no city is found, return None
    return None

def process_text2(text,exclude_words=[]):
    # Remove repeated characters
    text = re.sub(r'(\w)\1{2,}', r'\1', text)
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(w) for w in TextBlob(text).words])
    # Correct spelling
    #textblob = TextBlob(text)
    #exclude_words=['ikea','meatballs','meatball','ikeas']
    #for word in textblob.words:
    #    if word not in exclude_words:
    #       text = text.replace(word, str(TextBlob(word).correct()))    
    return text


import pandas as pd
import re

# Define the function to calculate the characters between the words

# View the resulting dataframe


#df_test['DistanceWords']             = df_test.apply(lambda row: calculate_characters_between_words2(row['Keyword'], row['Text']), axis=1)
df_test['Text_Lowercase']            = df_test['Text'].str.lower()

def calculate_characters(row):
    words = row['Keyword'].split(' ')
    text = row['Text_Lowercase']
    regex = '|'.join(words)
    matches = re.findall(regex, text)
    if len(matches) == 2:
        start_index = text.index(matches[0]) + len(matches[0])
        end_index = text.index(matches[1])
        characters_between = end_index - start_index
        return characters_between
    else:
        return None


#Creo una nueva columna para calcular los caracteres de diferencia que existen entre cada palabra de la busqueda de keyowrds
df_test['Characters_between_words'] = df_test.apply(calculate_characters, axis=1)

df_test['LocationCity']              = df_test['Text_Lowercase'].apply(lambda x: extract_city(x))
df_test['LocationCountry']           = df_test['Text_Lowercase'].apply(lambda x: extract_city(x))
df_test['Text_Lematized']            = df_test['filtered_list_3'].apply(process_text2)
df_test['Text_Stopword']             = df_test['Text_Lematized'].apply(remove_stopwords)



# Vectorize the text data using TF-IDF
X_test_tf_1 = tfidf.transform(df_test['Text_Stopword'])
X_test_tf_1 = X_test_tf_1.toarray().reshape(len(df_test), -1)

# Make predictions on the test data and obtain probability estimates
df_test['Polarity_Pred'] = calibrated_model.predict_proba(X_test_tf_1)[:, 1]


df_test.to_excel('TFM_ModeloSVM_14.xlsx',engine='xlsxwriter')
"""

job_config = bq.LoadJobConfig(
schema=[
    bq.SchemaField('id', 'STRING'),
    bq.SchemaField('created_at_short', 'Date'),
    bq.SchemaField('RelevantTweet', 'INT64'),
    bq.SchemaField('Author_id', 'STRING'),
    bq.SchemaField('User_Location', 'STRING'),
    bq.SchemaField('Retweet_Count', 'INT64'),
    bq.SchemaField('Like_Count', 'INT64'),
    bq.SchemaField('Replie_Count', 'INT64')	        ,
    bq.SchemaField('Language', 'STRING'),
    bq.SchemaField('Is_Ikea', 'STRING')	        ,
    bq.SchemaField('Keyword', 'STRING'),
    bq.SchemaField('Characters_between_words', 'INT64'),
   bq.SchemaField('Text', 'STRING'),    
    bq.SchemaField('English_Text', 'STRING')	,
	bq.SchemaField('Text_NotURL', 'STRING')     ,	
    bq.SchemaField('expanded_corpus', 'STRING')	,
    bq.SchemaField('filtered_list_2', 'STRING')	,
    bq.SchemaField('filtered_list_3', 'STRING')	,
    bq.SchemaField('Text_Lematized', 'STRING')	,
    bq.SchemaField('Polarity_Pred', 'FLOAT')	

    ]
)



table_id = 'ingka-food-analytics-dev.TFM_DEV_ingka_food_analytics_dev.Twitter_FullSentiment_Final'  #Nombre de la tabla
load_job = bq_client.load_table_from_dataframe(df_test, table_id, job_config=job_config) #Cambiar df_all por el nombre del dataframe que queráis cargar
load_job.result()



"""
