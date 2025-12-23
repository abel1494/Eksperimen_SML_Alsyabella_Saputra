import os
import re
import pickle
import warnings
import shutil
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# KONFIGURASI
SAMPLE_SIZE = 50000 
RAW_DATA_PATH = 'stacksample_raw'
OUTPUT_PATH = 'preprocessing/stacksample_preprocessing'

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
os.makedirs(OUTPUT_PATH, exist_ok=True)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_data():
    if not os.path.exists(f'{RAW_DATA_PATH}/Questions.csv'):
        os.system(f'kaggle datasets download -d stackoverflow/stacksample --unzip -p {RAW_DATA_PATH}')
    
    df_questions = pd.read_csv(f'{RAW_DATA_PATH}/Questions.csv', encoding='latin-1', nrows=SAMPLE_SIZE + 5000)
    df_tags = pd.read_csv(f'{RAW_DATA_PATH}/Tags.csv', encoding='latin-1')
    
    df_tags = df_tags[df_tags['Id'].isin(df_questions['Id'])]
    
    return df_questions, df_tags

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^a-zA-Z\s<>]', '', text)
    text = text.lower()
    words = text.split()
    clean_words = []
    for w in words:
        if w not in stop_words and len(w) > 2:
            clean_words.append(lemmatizer.lemmatize(w))
            
    return " ".join(clean_words)

def process_data(df_questions, df_tags):
    df_tags['Tag'] = df_tags['Tag'].astype(str)
    tags_grouped = df_tags.groupby('Id')['Tag'].apply(list).reset_index()
    
    df_final = pd.merge(df_questions, tags_grouped, on='Id', how='inner')
    df_final['text_raw'] = df_final['Title'] + " " + df_final['Body']
    
    df_final = df_final.dropna(subset=['text_raw'])
    df_final = df_final.drop_duplicates(subset=['text_raw'])

    df_final['text_clean'] = df_final['text_raw'].apply(clean_text)
    
    df_final = df_final[df_final['text_clean'].str.strip().astype(bool)]

    df_final['word_count'] = df_final['text_clean'].apply(lambda x: len(x.split()))
    df_final = df_final[df_final['word_count'] >= 5]

    cols_to_drop = [
    'Id', 'OwnerUserId', 'CreationDate', 'ClosedDate', 'Score', 
    'Title', 'Body', 'raw_char_length', 'text_raw', 'word_count']
    df_final.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    df_final = df_final.head(SAMPLE_SIZE)
    
    return df_final

def save_artifacts(df_final):
    all_tags = [tag for tags in df_final['Tag'] for tag in tags]
    top_tags = pd.Series(all_tags).value_counts().head(20).index.tolist()
    
    def filter_tags(tags):
        return [t for t in tags if t in top_tags]

    df_final['filtered_tags'] = df_final['Tag'].apply(filter_tags)
    df_final = df_final[df_final['filtered_tags'].str.len() > 0]
    
    mlb = MultiLabelBinarizer()
    mlb.fit(df_final['filtered_tags'])
    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    tfidf.fit(df_final['text_clean'])
    
    df_final[['text_clean', 'filtered_tags']].to_csv(f'{OUTPUT_PATH}/clean_data.csv', index=False)
    
    with open(f'{OUTPUT_PATH}/tfidf_model.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
    with open(f'{OUTPUT_PATH}/mlb_model.pkl', 'wb') as f:
        pickle.dump(mlb, f)
        
    print("[SUCCESS] Selesai! Script berjalan lancar.")

if __name__ == "__main__":
    questions, tags = load_data()
    final_df = process_data(questions, tags)
    save_artifacts(final_df)
