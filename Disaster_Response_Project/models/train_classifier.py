import sys
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    Loads data from sqlite database
    input:path to database file
    output: 
        X: features dataframe
        y: target dataframe
        category_names: names of targets
        
    '''
    # table name
    table_name = 'disaster_db'
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    df = pd.read_sql_table(table_name, engine)
   
    X = df['message'].values
   
    y = df.iloc[:, 4:]
    # get names 
    category_names = y.columns
    #print('category_names',category_names)
    return X, y, category_names


def tokenize(text):
    '''
    Function returns larger body of text into smaller lines, words or even creating words for a non-       English language.
    input : messages provided by the people for help in df dataframe
    '''
    # stopword list 
    STOPWORDS = list(set(stopwords.words('english')))
    # initialize lemmatier
    lemmatizer = WordNetLemmatizer()
    # split string into words (tokens)
    tokens = word_tokenize(text)
    
    tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    # remove stopwords
    tokens = [token for token in tokens if token not in STOPWORDS]
    # return data 
    #print('tokens',tokens[:5])
    return tokens


def build_model():
    '''Builds classification model'''
    # model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=100)))
    ])
    # hyper-parameter grid
    param_grid = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': [2, 3],
        'clf__estimator__n_estimators': [10, 15]
    }
   
    # create model 
    cv = GridSearchCV(pipeline, param_grid=param_grid, verbose=2, n_jobs=1, cv=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function will evaluate a  model with unseen dataset
    input: 
        model: trained model 
        X_test: Test features 
        Y_test: Test labels 
        category_names: names of lables
           
    '''
    # get predictions 
    y_pred = model.predict(X_test)
    # print classification report
    print(classification_report(Y_test,y_pred,  target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    '''
    Save the model to a Python pickle file 
    '''
    # save model binary 
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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