import sys
import pandas as pd
import numpy as np
import nltk 
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    '''
    Function reads the two csv files and provides the output of combined and processed dataframe
    Input: csv files
    output: merged dataframe
    '''
    # load messages data into dataframe
    messages = pd.read_csv(messages_filepath)
    # load categories data into dataframe
    categories = pd.read_csv(categories_filepath)
    # merge dataframes
    df = pd.merge(messages, categories, on="id")
  
    return df


def clean_data(df):
    '''
    Function cleans the raw data 
    iinput : merged dataframe from raw dataframe
    output : cleaned pandas dataframe
    '''
    # # create a dataframe of the 36 individual category columns
    s = pd.Series(df['categories'])
    s = s.str.split(';',expand=True)
    categories = pd.DataFrame(s)
    row = categories.iloc[0]
    category_colnames =[]
    #category_colnames = row.tolist()
    for item in row.tolist():
        item =item[:len(item)-2]
        category_colnames.append(item)
        
    
    #print('category_colnames',category_colnames)
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        #categories[column] = categories[column].str.slice(len(column)-1,len(column))
        #print('check',categories[column])
        #df['Date'].str[-4:].astype(int)
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] if int(x.split('-')[1]) < 2 else 1)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the categories column 
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new categories df
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    df.rename(columns={0:'related',1:'request',2:'offer',3:'aid_related',4:'medical_help',5:'medical_products',6:'search_and_rescue',7:'security',8:'military',9:'child_alone',10:'water',11:'food',12:'shelter',13:'clothing',14:'money',15:'missing_people',16:'refugees',17:'death',18:'other_aid',19:'infrastructure_related',20:'transport',21:'buildings',22:'electricity',23:'tools',24:'hospitals',25:'shops',26:'aid_centers',27:'other_infrastructure',28:'weather_related',29:'floods',30:'storm',31:'fire',32:'earthquake',33:'cold',34:'other_weather',35:'direct_report'},inplace=True)
    #print(df.columns)
    # return df
    return df


def save_data(df, database_filename):
    '''
    Function Saves pandas dataframe to database
    input: dataframe and file path
    output: (None)
    '''
   
    table_name = 'disaster_db'
    # create engine 
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # save dataframe to database, relace if already exists 
    df.to_sql(table_name, engine, index=False, if_exists='replace')


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