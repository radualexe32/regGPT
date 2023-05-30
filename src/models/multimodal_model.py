import torch
import torch.nn as nn
from text_model import *
from image_model import *

class MultimodalModel(nn.Module):
    def __init__(self, text_dimension, hidden_dimension, image_dimension, classes):
        super(MultimodalModel, self).__init__()
        # Fine tuning for models such as ResNet for image classification and BERT for text classification may be more useful than training from scratch.
        # The only left to do is to concatenate the outputs of the two models and feed them to a classifier. From there, it is possible to train the model. 
        # Type of data must be tailored specifically to the needs of the model. That is what the purpose of the fine tuning of these model is for.
        # To create a multimodal model that can accurately take in .csv files and data visualization images to model a linear regression line of best fit.
        self.text_model = nn.LSTM(input_size = text_dimension, hidden_size = hidden_dimension, num_layers = 1, batch_first = True)
        self.image_model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, image_dimension)
        )
        self.classfier = nn.Linear(hidden_dimension + image_dimension, classes)

    def forward(self, text, image):
        text_out, _ = self.text_model(text)
        image_out = self.image_model(image)
        text_out = text_out[:, -1, :]
        concat = torch.cat([text_out, image_out], dim = 1)
        output = self.classfier(concat)
        return output

if __name__ == "__main__":
    model = MultimodalModel(text_dimension = 300, hidden_dimension = 128, image_dimension = 256, classes = 10)
    text = torch.randn(64, 10, 300)
    image = torch.randn(64, 3, 64, 64)
    output = model(text, image)