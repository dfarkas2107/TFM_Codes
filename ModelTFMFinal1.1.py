import tensorflow_datasets as tfds
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from CONTRACTION_MAP import CONTRACTION_MAP
from sklearn.metrics import ConfusionMatrixDisplay
import re
from sklearn.dummy import DummyClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score,confusion_matrix
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
for example in ds:
#for example in ds.take(100000):
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


X_train, X_test, Y_train, Y_test = train_test_split(df['Text_Lematized'],df['polarity'],test_size=0.2,random_state=42,stratify=df['polarity'])
print('Size of Training Data ', X_train.shape[0])
print('Size of Test Data ', X_test.shape[0])

tfidf = TfidfVectorizer(min_df=5, ngram_range=(1, 6))
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

model1 = LinearSVC(random_state=0, 
                   tol=0.01,
                   C=1,
                   max_iter=10000,
                   penalty='l2',
                   loss='hinge'
                   )
model1.fit(X_train_tf, Y_train)
# Step 4 - Model Evaluation
X_test_tf = tfidf.transform(X_test)
Y_pred = model1.predict(X_test_tf)

from sklearn.metrics import precision_recall_fscore_support

print('Accuracy Score - ', accuracy_score(Y_test, Y_pred))

print('Precision Score TEST',precision_recall_fscore_support(Y_test, Y_pred, average='macro'))

print(classification_report(Y_test, Y_pred))

Y_pred_train = model1.predict(X_train_tf)

print('Accuracy Score (Training Data):', accuracy_score(Y_train, Y_pred_train))
print('Precision Score TRAIN',precision_recall_fscore_support(Y_train, Y_pred_train, average='macro'))

print('Classification Report (Training Data):\n', classification_report(Y_train, Y_pred_train))

#print('Accuracy Score - ', accuracy_score(Y_train, Y_pred))
#print(classification_report(Y_train, Y_pred))

clf = DummyClassifier(strategy='most_frequent')
clf.fit(X_train, Y_train)
Y_pred_baseline = clf.predict(X_test)
print ('Accuracy Score - ', accuracy_score(Y_test, Y_pred_baseline))
print ('Accuracy Score - ', precision_recall_fscore_support(Y_test, Y_pred_baseline))

print(confusion_matrix(Y_test, Y_pred))

