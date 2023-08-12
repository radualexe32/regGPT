import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class DecisionTree(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.linear(x)
        return F.sigmoid(out)

    def predict(self, x):
        out = self.forward(x)
        return (out > 0.5).float()

    def fit(self, x, y, epochs=100, learning_rate=0.01):
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            inputs = Variable(torch.from_numpy(x))
            targets = Variable(torch.from_numpy(y))

            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
