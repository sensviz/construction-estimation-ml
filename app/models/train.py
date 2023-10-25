import torch
from torch.nn import Linear, MSELoss
from sklearn.metrics import mean_squared_error
from torch.optim import SGD
from sklearn.model_selection import train_test_split
import pandas as pd

def train(data):
     # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('cost', axis=1), data['cost'], test_size=0.2, random_state=42)
    # Convert the data to PyTorch tensors
    X_train = torch.FloatTensor(X_train.values)
    y_train = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test = torch.FloatTensor(X_test.values)
    y_test = torch.FloatTensor(y_test.values).view(-1, 1)
    # Create the linear regression model
    model = Linear(X_train.shape[1], 1)
    # Define the loss function and optimizer
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    # Train the model
    for epoch in range(1000):
        # Forward pass
        outputs = model(X_train)
        # Compute the loss
        loss = criterion(outputs, y_train)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Zero the gradients
        optimizer.zero_grad()
        # Check for overfitting
        if epoch % 100 == 0:
            with torch.no_grad():
                y_pred = model(X_test)
                loss = criterion(y_pred, y_test)
                if loss.item() > 0.1:
                    print('Overfitting detected. Stopping training.')
                    break
    
    torch.save(model.state_dict(), 'model.pt')
    y_pred = model(torch.from_numpy(X_test).float())
    mse = mean_squared_error(y_test, y_pred.detach().numpy())
    print(f'MSE: {mse}')