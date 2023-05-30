# Here we use the ResNet (or something else) model from the transformers library of HuggingFace that serves as the image model for the multimodal model.
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")