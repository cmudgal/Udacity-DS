# Load and process data
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
     Function: This function reads and loads data from two csv files and merges them.
     
     Arguments: messages_filepath(str)- path to messages.csv
                categories_filepath(str)- path to categories.csv
               
     Return: returns dataframe of messages and categories
     
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on='id')
    
    return df


def clean_data(df):
    """
     Function: This function splits the categories based on                the token and  cleans the df.
     
     Arguments: dataframe of merged messages and categories
               
     Return: returns clean dataframe of messages and categories
     
    """
    # Split `categories` into separate category columns.
    categories = df['categories'].str.split(';', expand=True)

   
    # select the first row of the categories dataframe
    row = categories.iloc[0].values
    category_colnames = [r[:-2] for r in row]

    
    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
    # set each value to be the last character of the string
    
     categories[column] = categories[column].str[-1]
     categories[column] = pd.to_numeric(categories[column])
    
    # show the list
     print( categories[column].values)
    
    categories.head()
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')

    # Drop the duplicates.
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
       Function: Save the Dataframe df in a database
       Arguments:df (DataFrame): A dataframe of messages and categories dataset
       database_filename (str): The file name of the database
       
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages_tbl', engine, index=False, if_exists = 'replace')
     


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
