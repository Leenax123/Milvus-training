from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
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
sentence_C_vector: List[float] = hf.embed_query("Tomorrow will be raining")

# Compute similarity

similarityA_B = float(util.pytorch_cos_sim(sentence_A_vector, sentence_B_vector)) # 0.55
similarityA_C = float(util.pytorch_cos_sim(sentence_A_vector, sentence_C_vector)) # 0.07

# Le modele a bien repere la similarite entre A & B, par rapport a C.
