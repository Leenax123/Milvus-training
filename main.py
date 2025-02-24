from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

# Choose a model
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

# Load Embeddor engine
hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Embed sentence

sentence_A_vector: List[float] = hf.embed_query("I like eating banana")
sentence_B_vector: List[float] = hf.embed_query("Fruits are so good")
