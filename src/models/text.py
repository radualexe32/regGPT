# Here we use the BERT (or something else) model from the transformers library of HuggingFace that serves as the text model for the multimodal model.
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-cased")