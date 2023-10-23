import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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


   





