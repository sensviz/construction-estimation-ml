import torch
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from helpers.preprocess import label_encode_categorical_values,one_hot_encoding

def test():
    
    # Load the data
    data = pd.read_excel('cost_estimation.xlsx')
    data=label_encode_categorical_values(data)
    data=one_hot_encoding(data)
    data.to_csv('test_raw_data.csv', index=False) 

    y_test=data['cost']
    X_test =data.drop('cost', axis=1)

    # Convert the data to PyTorch tensors
    X_test = torch.FloatTensor(X_test.values)
    y_test = torch.FloatTensor(y_test.values)
    model = Linear(X_test.shape[1], 1)
    model.load_state_dict(torch.load('model.pt'))
    # Make predictions
    y_pred = model(X_test)
    print(y_pred)
    # Evaluate the model
    # mse = mean_squared_error(y_test, y_pred.detach())
    # print(f'MSE: {mse}')