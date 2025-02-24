import requests
from pymilvus import Collection

# Connexion à Milvus
milvus_client = Collection("collection") 

query_result = milvus_client.query(expr="id == 1")   

vector_to_send = query_result[0]["vector"]  # Extraire le vecteur


url = "http://127.0.0.1:8000/search/"
payload = {"vector": vector_to_send}
response = requests.post(url, json=payload)

# Affichage de la réponse de l'API
print(response.json())