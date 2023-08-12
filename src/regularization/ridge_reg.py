import torch
import torch.nn as nn
from torch.nn import functional as F


class RidgeRegression(nn.Module):
    def __init__(self, input_dim, output_dim, lambda_param):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.lambda_param = lambda_param

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        return self.lambda_param * torch.sum(self.linear.weight**2)

    def total_loss(self, output, target):
        mse_loss = F.mse_loss(output, target, reduction="sum")
        return mse_loss + self.regularization_loss()
