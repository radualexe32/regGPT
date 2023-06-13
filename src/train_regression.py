from sklearn.datasets import make_regression
import torch
from torch.utils.data import TensorDataset, DataLoader
from reg import Regression
from train import data 
import matplotlib.pyplot as plt

n_samples, n_features = 100, 1
X, y = make_regression(n_samples = n_samples, n_features = n_features, noise = 5)
y = y ** 2

plt.scatter(X[:, 0], y)
plt.show()

X_tensor = torch.tensor(X, dtype = torch.float32)
y_tensor = torch.tensor(y, dtype = torch.float32).unsqueeze(1)  

dataset = TensorDataset(X_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size = 32)
model = Regression(n_features, 1, regression_type = "polynomial", degree = 2)
model.train(data_loader, epochs = 1000)
