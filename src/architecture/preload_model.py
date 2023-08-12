# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ")
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ")
# TODO: Fix the error regarding loading the model
