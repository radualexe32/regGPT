import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class RandomForest(nn.Module):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth
        )

    def forward(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y):
        pred = self.model.predict(x)
        mse = mean_squared_error(y, pred)
        return mse
