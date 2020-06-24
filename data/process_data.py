# import libraries
import pandas as pd
import sys
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # Remove duplicates
    messages.drop_duplicates(inplace = True)
    categories.drop_duplicates(inplace = True)
   
    # Join tables
    df = messages.merge(categories, on = 'id', how = 'left')
    
    return df

def clean_data(df):
    # 1. Split categories into separate category columns (create a dataframe of the 36 individual category columns)
    categories = df.loc[:,'categories'].str.split(';', expand = True)
    # 2. Name columns with appropriate name
    category_colnames = [i.split('-')[0] for i in categories.iloc[0, :]]
    # 3. Rename the columns of `categories`
    categories.columns = category_colnames
    # 4. Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        var = categories[column].str.split('-')
        categories[column] = [item[1] for item in var]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    # 5. Replace categories column in df with new category columns
        # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
        # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine, index=False, if_exists = 'replace')  


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