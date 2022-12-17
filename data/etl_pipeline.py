import sys
# import libraries
import re
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

import nltk
nltk.download(['stopwords', 'wordnet'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def load_data(filepath):
    '''
    DESCRIPTION
        load content of the file defined by filepath and convert it to dataframe
    INPUT
        filepath is the filepath of the file to load
    OUTPUT
        df is the dataframe of the file's content
    '''
    df = pd.read_csv(filepath)
    return df


def merge_data(df1, df2):
    '''
    DESCRIPTION
        Merge both dataframes messages and categories by their 'id' column
    INPUT
        df1 is the first dataframe that will be completed with the second
        df2 is the second dataframe to complete the first one
    OUTPUT
        df1 is the merge of both dataframes, sorted by the id values
    '''
    df = pd.merge(df1, df2, how='outer', on=['id'])
    df.sort_values(['id'])
    return df


def dummy_data(df):
    '''
    DESCRIPTION
        add dummy data to df removing the source of dummy
    INPUT
        df is the initial dataframe to work on
    OUTPUT
        df_dummy is the dataframe as copy of df but with dummy columns, removing the source of dummy
    '''       
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.head(1)  # categories.values[:1][0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda name: name.str.split('-')[0][0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda name: name[-1:])        
        # convert column from string to numeric
        # categories[column] = categories[column].astype(int)  # is replaced by
        categories[column] = pd.get_dummies(categories[column])
    
    # Replace categories column in df with new category columns
    # by dropping the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_dummy = pd.concat([df, categories], axis=1, join='outer')
    
    # Remove duplicates
    # - check number of duplicates
    duplicates = df_dummy.duplicated().sum()
    if duplicates > 0:
        df_dummy.drop_duplicates(keep='first', inplace=True)

    return df_dummy


def clean_data(df):
    '''
    DESCRIPTION
        clean dataframe from empty data
    INPUT
        df is the dataframe to clean
    OUTPUT
        df_clean is the cleaned dataframe
    '''   
    # Clean the dataframe
    df_clean = df.dropna(how='any')  # remove rows with an empty field 
    return df_clean
    

def save_data(df, database_filepath):
    '''
    DESCRIPTION
        save a dataframe into database file
    INPUT
        df is the dataframe to save
        database_filepath if the filepath of the database, i.e. target of the save
    OUTPUT
        nil
    '''
    table_name = 'DisasterResponse'
    # conn = sqlite3.connect(database_filepath)
    # df.to_sql(table_name, conn, if_exists='replace', index=False)
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql(table_name, engine, if_exists='replace', index=False)
 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df_msg = load_data(messages_filepath)
        df_cat = load_data(categories_filepath)

        print('Data Merge...')
        df = merge_data(df_msg, df_cat)
        
        print('Data Merge...')
        df = dummy_data(df)
        
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