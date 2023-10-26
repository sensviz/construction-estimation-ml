import torch
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss
from sklearn.model_selection import train_test_split
import pandas

from sklearn.metrics import mean_squared_error
# from helpers.preprocess import label_encode_categorical_values,one_hot_encoding

def test(data):
    
    # Load the data 

    # y_test=data[variable]
    # X_test =data.drop(variable, axis=1)

    # Convert the data to PyTorch tensors
    X_test = torch.FloatTensor(data.values)
    y_test = torch.FloatTensor(y_test.values)
    model = Linear(X_test.shape[1], 1)
    model.load_state_dict(torch.load('model.pt'))
    # Make predictions
    y_pred = model(X_test)
    return y_pred
    # Evaluate the model
    # mse = mean_squared_error(y_test, y_pred.detach())
    # print(f'MSE: {mse}')