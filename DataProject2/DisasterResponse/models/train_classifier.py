# ML pipeline
import sys
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pickle

nltk.download(['wordnet', 'punkt', 'stopwords'])

def load_data(database_filepath):
    """
       Function:
       load data from database
       Arguments:
       database_filepath: the path of the database
       Return:
       X (DataFrame) : Message features dataframe
       Y (DataFrame) : target dataframe
       category (list of str) : target labels list
     """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages_tbl', engine)
    X = df['message']  # Message Column
    Y = df.iloc[:, 4:]  # Classification label
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns
    return X, Y, category_names
   


def tokenize(text):
    """
     Function: split text into words and return words in root form
     Args:
      text(str): the message
     Return:
      lemm(list of str): a list of the root form of the message words
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') 
             for word in words]
    
    return words
    


def build_model():
    """
     Function: build a model for classifing the disaster messages
     Return:
       Pipeline
     """
    pipeline =  Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())))])
    return pipeline
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
      Function: Evaluate the model and print the f1 score, precision and recall for each output category of the dataset.
      Arguments: 
      model: the classification model
      X_test: test messages
      Y_test: test target
      category_names: Categories
   """
    y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))
    

def improve_model(X_train, y_train, X_test, y_test, model):
    # hyper-parameter grid
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create model
    cv = GridSearchCV(estimator=model,
            param_grid=parameters,
            verbose=3,
            cv=3)

    cv.fit(X_train, y_train)

    # test improved model
    optimised_model = cv.best_estimator_
    print (cv.best_estimator_)
    y_pred = optimised_model.predict(X_test)

    print(classification_report(y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    """
     Function: Save a pickle file of the model
     Args:
     model: the classification model
     model_filepath (str): the path of pickle file
   """
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        print('Building model...')
        model = build_model()
        print('Training model...')
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        improve_model(X_train, y_train, X_test, y_test, model)
        
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
