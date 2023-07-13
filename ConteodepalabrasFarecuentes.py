import tensorflow_datasets as tfds
import pandas as pd
import re
import numpy as np
from CONTRACTION_MAP import CONTRACTION_MAP
import re
from nltk.corpus import stopwords
import pandas as pd
from nltk import pos_tag, FreqDist
from nltk.tokenize import word_tokenize
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

license_file_path = "C:/Users/diego/Downloads/ingka-food-analytics-dev-2127cf9baa38.json"
credentials = service_account.Credentials.from_service_account_file(license_file_path)
bq_client = bq.Client(credentials=credentials, project=credentials.project_id)
# Example dataframe

query_string = '''
SELECT Keyword,English_Text FROM  `ingka-food-analytics-dev.TFM_DEV_ingka_food_analytics_dev.Twitter_FullSentiment_Final`  where Keyword is not null
and Is_Publication=1 AND RN=1 and RelevantTweet=1  
;'''
df = bq_client.query(query_string).to_dataframe()


# Function to count adjectives in a sentence
def count_adjectives(sentence):
    tokens = word_tokenize(sentence)
    return len(tokens)
# Apply the function to Column2, excluding the words from Column1
df['Adjectives'] = df.apply(lambda row: count_adjectives(row['English_Text'].replace(row['Keyword'].split()[1], '')), axis=1)

# Apply the function to Column2, excluding the words from Column1

# Group by Column1 and calculate the frequency distribution for each group
grouped_data = df.groupby('Keyword')['Adjectives'].apply(lambda x: [adj for sublist in x for adj in sublist])
grouped_freq_dist = grouped_data.apply(lambda x: FreqDist(x))

# Get the most frequent adjectives for each group
most_frequent_adjectives = grouped_freq_dist.apply(lambda x: [adj for adj, _ in x.most_common(50)])
result_df = pd.DataFrame({'Column1': most_frequent_adjectives.index, 'Most Frequent Adjectives': most_frequent_adjectives.values})

print(result_df)
result_df.to_excel('adjectives2.xlsx',engine='xlsxwriter')
