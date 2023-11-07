import torch
from torch.utils.data import DataLoader
from torch.nn import Linear, MSELoss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from app.models.train import LinearRegression

def test(data):
    print(data.shape)
    X = data.drop('Cost', axis = 1)
    y = data['Cost']
    y = data.values
    print(X.shape)
    X = X.values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    # Load the model
    model = LinearRegression(input_dim=X.shape[1], hidden_dim=50, output_dim=1)  # Make sure to set the correct input and hidden dimensions

    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    return predictions
