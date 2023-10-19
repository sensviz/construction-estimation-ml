import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

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



