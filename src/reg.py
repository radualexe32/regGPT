import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

class Regression(nn.Module):
    def __init__(self, input_dim, output_dim, regression_type, degree = None):
        super(Regression, self).__init__()

        self.regression_type = regression_type
        self.linear = nn.Linear(input_dim, output_dim)

        if regression_type == "polynomial":
            self.degree = degree
            self.poly = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(degree)])
        elif regression_type == "logistic":
            self.sigmoid = nn.Sigmoid()

        if self.regression_type == "logistic":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = optim.SGD(self.parameters(), lr = 0.01)

    def forward(self, x):
        if self.regression_type == "polynomial":
            x = sum([layer(x ** i) for i, layer in enumerate(self.poly, start = 1)])
        else:
            x = self.linear(x)
        
        if self.regression_type == "logistic":
            x = self.sigmoid(x)
            
        return x
    
    def train_step(self, inputs, targets):
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            for inputs, targets in data_loader:
                loss = self.train_step(inputs, targets)

            if epoch == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1} / {epochs} Loss: {loss}")

        with torch.no_grad():
            predictions = []
            true_values = []
            for inputs, targets in data_loader:
                outputs = self(inputs)
                predictions.extend(outputs.numpy())
                true_values.extend(targets.numpy())

            mse_after = mean_squared_error(true_values, predictions)
            r2_after = r2_score(true_values, predictions)

        print("After training:")
        print(f"MSE: {mse_after}")
        print(f"R2: {r2_after}")
        print(f"Weights: {self.state_dict()}")

 

