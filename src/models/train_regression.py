from imports import *
from regGPT import Regression
import matplotlib.pyplot as plt


def train(reg_type="linear", degree=1, graph=True):
    n_samples, n_features = 100, 1
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=5)
    y = y ** degree

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = Regression(n_features, 1, regression_type=reg_type, degree=degree)
    model.train(train_loader, val_loader, epochs=1000)

    x_line = np.linspace(X_train.min(), X_train.max(), 500)
    x_line_tensor = torch.tensor(x_line, dtype=torch.float32).unsqueeze(1)
    y_line = model(x_line_tensor).detach().numpy()

    if graph:
        # Plot the data and the model's line
        plt.figure(figsize=(10, 5))
        plt.scatter(X_train, y_train)
        plt.plot(x_line, y_line, color='r', label='reg line')
        plt.legend()
        plt.show()

    return model
