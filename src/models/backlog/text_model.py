from transformers import pipeline
import openai

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
openai.api_key = "XXX-XXX-XXX"

var = "housing prices in USD"
sequence = "The independent variable is " + var

res = classifier(
    sequence,
    candidate_labels=["categorical", "numerical"]
)

# res = openai.ChatCompletion.create(
#    model = "gpt-3.5-turbo",
#    messages = [
#        {"role": "system", "content": "You are classifyGPT. You only output whether a variable is categorical or numerical."},
#        {"role": "user", "content": sequence},
#    ]
# )

print(res)
print(res["choices"][0]["message"]["content"])
