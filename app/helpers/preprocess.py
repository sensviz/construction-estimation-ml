import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess():
    df = pd.read_csv("cost_estimation.csv")
    label_encoder = OneHotEncoder(sparse=False)
    print(df.head(8))
    print(df.columns)
    if df.isnull().values.any():
        raise ValueError("DataFrame contains null values. Please handle the null values before proceeding.")

    transformer = ColumnTransformer([
    ('tnf1',OneHotEncoder(sparse=False,dtype=np.int32),['material']),
    ('tnf2',OneHotEncoder(sparse=False),['product'])
],remainder='passthrough')
    
    df_encoded =  transformer.fit_transform(df)
    print(df_encoded.shape)
    df2=pd.DataFrame(df_encoded,columns=['concrete hexagon pavors','exterior perforated metal panel','exterior metal penal','exterior glass-fibre reinforced concrete faÃ§ade panels','field paint','meeting room','toilet', 'drawing room', 'living', 'id', 'cost', 'quantity',
       'area'])
    
    return df2

def label_encode_categorical_values(X):
    label = LabelEncoder()
    X['id'] = label.fit_transform(X.id.values)
    return X

def one_hot_encoding(X):
    one_hot_encoded = pd.get_dummies(X)
    X=one_hot_encoded
    return X

# Load the data
data = pd.read_excel('cost_estimation.xlsx')

data=label_encode_categorical_values(data)
data=one_hot_encoding(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('cost', axis=1), data['cost'], test_size=0.2, random_state=42)
# Convert the data to PyTorch tensors
X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values).view(-1, 1)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.FloatTensor(y_test.values).view(-1, 1)



