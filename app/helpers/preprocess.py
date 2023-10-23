import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
encoder = OneHotEncoder()
columns_to_encode = ['product', 'material']

def label_encode_categorical_values(X):
    label = LabelEncoder()
    X['id'] = label.fit_transform(X.id.values)
    return X

def one_hot_encoding(X):
    one_hot_encoded = pd.get_dummies(X)
    X=one_hot_encoded
    return X

def preprocess(): 

    # Load the data
    data = pd.read_excel('cost_estimation.xlsx')
    data=label_encode_categorical_values(data)
    data=one_hot_encoding(data)
    data.to_csv('preprocessed_data.csv', index=False) 



def preprocessEncoding():   
    df = pd.read_csv("cost_estimation.csv")
    if df.isnull().values.any():
        raise ValueError("DataFrame contains null values. Please handle the null values before proceeding.")

    print(df.columns)
    encoded_columns = encoder.fit_transform(df[columns_to_encode]).toarray()

# Creating a new DataFrame with the encoded columns
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(columns_to_encode))
    df = df.drop(columns_to_encode , axis = 1)
    df_encoded = pd.concat([df, encoded_df], axis=1)
    print(df_encoded.head())
    
    return df_encoded

def preprocessDecoding(df):
    columns = ['id','cost','quantity', 'area']
    new_df = df[columns]
    encode_df = df.drop(columns,axis = 1)
    decoded_columns = encoder.inverse_transform(df[encode_df.columns])
    df_decoded = pd.DataFrame(decoded_columns, columns=columns_to_encode)
    
    print(decoded_columns)
    print(df_decoded)
    df_decoded = pd.concat([df.drop(encode_df.columns, axis=1), df_decoded], axis=1)
    print(df_decoded)
    return df_decoded
