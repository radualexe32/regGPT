import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


class Regression(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, regression_type="linear", degree=None):
        super(Regression, self).__init__()

        self.regression_type = regression_type

        if regression_type == "polynomial":
            self.degree = degree
            self.poly = nn.ModuleList(
                [nn.Linear(input_dim, output_dim) for _ in range(degree)])
        elif regression_type == "logistic":
            self.linear = nn.Linear(input_dim, output_dim)
            self.sigmoid = nn.Sigmoid()
        else:
            self.linear = nn.Linear(input_dim, output_dim)

        if self.regression_type == "logistic":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=10)
        self.mse, self.r2, self.correlation = [None for _ in range(3)]

    def forward(self, x):
        if self.regression_type == "polynomial":
            x = sum([layer(x ** i)
                    for i, layer in enumerate(self.poly, start=1)])
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

    # def train(self, data_loader, epochs):
    #     for epoch in range(epochs):
    #         for inputs, targets in data_loader:
    #             loss = self.train_step(inputs, targets)

    #         if epoch == 0 or epoch == epochs - 1:
    #             print(f"Epoch {epoch + 1} / {epochs} Loss: {loss}")

    #     with torch.no_grad():
    #         predictions = []
    #         true_values = []
    #         for inputs, targets in data_loader:
    #             outputs = self(inputs)
    #             predictions.extend(outputs.numpy())
    #             true_values.extend(targets.numpy())

    #         mse_after = mean_squared_error(true_values, predictions)
    #         r2_after = r2_score(true_values, predictions)
    #         self.mse, self.r2 = mse_after, r2_after

    #     print("After training:")
    #     print(f"MSE: {mse_after}")
    #     print(f"R2: {r2_after}")
    #     print(f"Weights: {self.state_dict()}")

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                loss = self.train_step(inputs, targets)

            with torch.no_grad():
                preds = []
                true_vals = []
                val_loss = 0
                for inputs, targets in val_loader:
                    outputs = self(inputs)
                    val_loss += self.criterion(outputs, targets).item()
                    preds.extend(outputs.numpy().flatten())
                    true_vals.extend(targets.numpy().flatten())

                # Average the validation loss
                val_loss /= len(val_loader)

            # Step the scheduler
            self.scheduler.step(val_loss)

            if epoch == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch + 1} / {epochs} Loss: {loss} Validation Loss: {val_loss}")

                mse_after = mean_squared_error(true_vals, preds)
                r2_after = r2_score(true_vals, preds)
                corr, _ = pearsonr(true_vals, preds)
                self.mse, self.r2, self.correlation = mse_after, r2_after, corr

    def get(self):
        return self.state_dict()

    def get_mse(self):
        return self.mse

    def get_r2(self):
        return self.r2

    def get_correlation(self):
        return self.correlation
