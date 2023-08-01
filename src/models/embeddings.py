import pandas as pd
from transformers import AutoTokenizer, AutoModel
from chromadb.api.types import EmbeddingFunction


class CSVEmbedding(EmbeddingFunction):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")

    def __call__(self, file, columns):
        df = pd.read_csv(file)
        embeddings_dict = {}

        # Iterate through each column
        for column in columns:
            docs = df[column].tolist()

            embeddings = []
            for doc in docs:
                input = self.tokenizer(str(doc), return_tensors="pt")
                output = self.model(**input)

                # Append tokenized data into vector db
                embeddings.append(
                    output.last_hidden_state[0].mean(dim=0).detach().numpy())

            embeddings_dict[column] = embeddings

        return embeddings_dict
