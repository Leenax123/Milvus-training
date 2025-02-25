from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
from typing import List
from fastapi import FastAPI
import uvicorn

app = FastAPI()
# Choose a model
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

# Load Embeddor engine
hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


@app.get("/similarity")
def compute_similarity(sentence_a, sentence_b):

    # Embed sentences
    sentence_A_vector: List[float] = hf.embed_query(sentence_a)
    sentence_B_vector: List[float] = hf.embed_query(sentence_b)

    # Compute similarity
    similarity = float(util.pytorch_cos_sim(sentence_A_vector, sentence_B_vector))

    return {"similarity": similarity}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
