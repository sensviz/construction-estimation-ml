import torch
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear, MSELoss
from sklearn.metrics import mean_squared_error
from torch.optim import SGD
from sklearn.model_selection import train_test_split
import pandas as pd


# # Define a simple regression model
# class RegressionModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(RegressionModel, self).__init__()
#         self.linear = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         return self.linear(x)

class LinearRegression(nn.Module):
  def __init__(self, input_dim: int, 
               hidden_dim: int, output_dim: int) -> None:
    super(LinearRegression, self).__init__()
    self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
    self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
    self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
    self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.input_to_hidden(x)
    x = self.hidden_layer_1(x)
    x = self.hidden_layer_2(x)
    x = self.hidden_to_output(x)
    return x

def train(data  , epoch , variable , split):
    print(data.tail(1))
    print(data.dtypes)
    # Convert non-numeric columns to numeric
    for column in data.columns:
        if data[column].dtype == np.object_:
            data[column] = pd.to_numeric(data[column], errors='coerce')
     # Split the data into training and testing sets
    split = split/100
    X_train, X_test, y_train, y_test = train_test_split(data.drop(variable, axis=1), data[variable], test_size=split, random_state=42)
    X = X_train.values
    y = y_train.values
    print(X.shape)

    # # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    # Reshape the tensors to have a single batch dimension
    # X_tensor = X_tensor.unsqueeze(1)
    # y_tensor = y_tensor.unsqueeze(1)
    input_size = X.shape[1]
    output_size = 1
       # Convert the data to PyTorch tensors
    # Create the linear regression model
    model = LinearRegression(input_size,50, output_size)
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)

    # Training the model
    num_epochs = epoch
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model.pt')
    X = X_test.values
    y = y_train.values

    # # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    # Reshape the tensors to have a single batch dimension
    # X_tensor = X_tensor.unsqueeze(1)
    # y_tensor = y_tensor.unsqueeze(1)
    with torch.no_grad():
        predicted = model(X_tensor)
    predicted_np = predicted.numpy()
    y_test = y_tensor.numpy()
    # Convert y_tensor to a numpy array
    mse_loss = np.mean((predicted_np - y_test) ** 2)
    print(f'Mean Squared Error (MSE): {mse_loss:.4f}')
    return mse_loss
