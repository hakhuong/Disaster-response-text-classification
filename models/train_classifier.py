# import libraries
import pandas as pd
import re 
import sys
import pickle
# database
from sqlalchemy import create_engine
import sqlite3
# NLP
import nltk
nltk.download('punkt') 
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

# ML
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# Tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
#Customize transformer
from sklearn.base import BaseEstimator, TransformerMixin
# Validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report



def load_data(database_filepath):
    # load data from database
    conn = sqlite3.connect(database_filepath)

    df = pd.read_sql('SELECT * FROM df', conn)
    df = df[df['related']!=2] # Delete value 2 in related column
    X = df.loc[:, 'message'].copy()

    y = df.iloc[:, 4:].copy()
    y = y.drop(columns = ['child_alone'], axis = 1) # because there is no observation
    
    category_names = y.keys() 
    return X, y, category_names

def tokenize(text):
    '''
    INPUT: text in string format
    OUTPUT: text in list format, tokenized, removed stop words, and lemmatized
    '''
    # Replace URL with 'urlplaceholder'
    #### Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    #### Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    #### Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'urlplaceholder')
        
    # Normalization
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text) # remove punctuation & number
    
    # Tokenization
    text = word_tokenize(text)
    
    # Stop word removal 
    text = [word for word in text if word not in stopwords.words('english')]
    
    # Lemmatization
    text = [WordNetLemmatizer().lemmatize(word) for word in text]

    return text

# Create textLengthExtractor()
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    
    def count_length(self, text):
        text_length = len(text)
        return text_length
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        text_length = pd.Series(X).apply(self.count_length)
        return pd.DataFrame(text_length)
        

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                    ('vec', CountVectorizer(tokenizer = tokenize)), 
                    ('tfidf', TfidfTransformer())
            ])),
            ('txt_len', TextLengthExtractor())

        ])), 
        ('scale', StandardScaler(with_mean=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline




def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()