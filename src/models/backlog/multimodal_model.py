import torch
import torch.nn as nn
from text_model import *
from image_model import *


class MultimodalModel(nn.Module):
    def __init__(self, text_dimension, hidden_dimension, image_dimension, classes):
        super(MultimodalModel, self).__init__()

        self.text_model = nn.LSTM(
            input_size=text_dimension, hidden_size=hidden_dimension, num_layers=1, batch_first=True)
        self.image_model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, image_dimension)
        )
        self.classfier = nn.Linear(hidden_dimension + image_dimension, classes)

    def forward(self, text, image):
        text_out, _ = self.text_model(text)
        image_out = self.image_model(image)
        text_out = text_out[:, -1, :]
        concat = torch.cat([text_out, image_out], dim=1)
        output = self.classfier(concat)
        return output


if __name__ == "__main__":
    model = MultimodalModel(
        text_dimension=300, hidden_dimension=128, image_dimension=256, classes=10)
    text = torch.randn(64, 10, 300)
    image = torch.randn(64, 3, 64, 64)
    output = model(text, image)
