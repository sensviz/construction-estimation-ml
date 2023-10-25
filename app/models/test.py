import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def test():
    # Load the model
    model = torch.load('model.pt')
    # Load the data
    data = pd.read_csv('preprocessed_data.csv')
     # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('cost', axis=1), data['cost'], test_size=0.0, random_state=42)
    # Convert the data to PyTorch tensors
    X_train = torch.FloatTensor(X_train.values)
    
    # Make predictions
    y_pred = model(torch.from_numpy(X_train).float())
    # Evaluate the model
    mse = mean_squared_error(y_train, y_pred.detach().numpy())
    print(f'MSE: {mse}')