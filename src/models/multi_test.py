import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from regGPT import Regression

n_samples = 1000
n_features = 3
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1)

# Rescale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_val = torch.from_numpy(X_val.astype(np.float32))
y_val = torch.from_numpy(y_val.astype(np.float32))

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_data, batch_size=32)
val_loader = DataLoader(val_data, batch_size=32)

model = Regression(input_dim=n_features, output_dim=1, regression_type="multi")

# Train the model
model.train(train_loader, val_loader, epochs=100)

# Print the model's performance metrics
print(f"MSE: {model.get_mse()}")
print(f"R2: {model.get_r2()}")
print(f"Correlation: {model.get_correlation()}")
