import tensorflow_datasets as tfds
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from CONTRACTION_MAP import CONTRACTION_MAP
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
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

ds = tfds.load('sentiment140', split='train')
data = []
#for example in ds:
for example in ds.take(500000):
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


from transformers import pipeline

training_pipeline = Pipeline(
  steps=[('tfidf', TfidfVectorizer(stop_words="english")),
        ('model', LinearSVC(random_state=42, tol=1e-5))])

grid_param = [{
'tfidf__min_df': [5, 10],
'tfidf__ngram_range': [(1, 3), (1, 6)],
'model__penalty': ['l2'],
'model__loss': ['hinge'],
'model__max_iter': [1000,10000],
'model__C': [1, 10],
'model__tol': [1e-2, 1e-3]
}, {
'tfidf__min_df': [5, 10],
'tfidf__ngram_range': [(1, 3), (1, 6)],
'model__C': [1, 10],
'model__tol': [1e-2, 1e-3]
}]
gridSearchProcessor = GridSearchCV(estimator=training_pipeline,param_grid=grid_param,cv=5)

gridSearchProcessor.fit(df['Text_Lematized'], df['polarity'])
best_params = gridSearchProcessor.best_params_
print("Best alpha parameter identified by grid search ", best_params)
best_result = gridSearchProcessor.best_score_
print("Best result identified by grid search ", best_result)


gridsearch_results = pd.DataFrame(gridSearchProcessor.cv_results_)
gridsearch_results[['rank_test_score', 'mean_test_score','params']].sort_values(by=['rank_test_score'])[:5]
gridsearch_results.to_excel('GridSearch.xlsx',engine='xlsxwriter')

print(gridsearch_results)