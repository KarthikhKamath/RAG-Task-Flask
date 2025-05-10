import requests
import trafilatura
from sentence_transformers import SentenceTransformer
from uuid import uuid4
import json
import chromadb
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

storage_path = os.path.join(os.getcwd(), "chroma_store")
if not os.path.exists(storage_path):
    os.makedirs(storage_path)

client = chromadb.PersistentClient(path=storage_path)


try:
    collection = client.get_collection("news_articles")
except chromadb.errors.NotFoundError:
    collection = client.create_collection("news_articles")

# NewsAPI key
api_key = 'c959b631349546a1b1fcf522d8fd4d63'
url = f'https://newsapi.org/v2/top-headlines?category=technology&apiKey={api_key}&pageSize=100&language=en'

response = requests.get(url)
documents = []

# Function to split content and handle small paragraphs
def chunk_content(content):
    paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 0] 
    final_paragraphs = []
    temp = ""

    for p in paragraphs:
        if len(p) < 200: 
            temp += " " + p
        else:
            if temp:
                final_paragraphs.append(temp) 
                temp = "" 
            final_paragraphs.append(p)  

    if temp: 
        final_paragraphs.append(temp)
    
    return final_paragraphs

if response.status_code == 200:
    data = response.json()
    articles = data['articles']
    print("Extracting and embedding content...\n")

    print(f"Number of articles fetched: {len(articles)}")

    for idx, article in enumerate(articles[:50]):
        article_url = article['url']
        downloaded = trafilatura.fetch_url(article_url)

        

        if downloaded:
            content = trafilatura.extract(downloaded)
            if content:
                paragraphs = chunk_content(content)  # Split and merge small paragraphs
                for paragraph in paragraphs:
                    embedding = model.encode(paragraph, normalize_embeddings=True)
                    document = {
                        "id": str(uuid4()),  # Unique ID for each chunk
                        "url": article_url,
                        "text": paragraph,
                        "embedding": embedding.tolist() 
                    }
                    
                    # Upsert into Chroma (store embeddings and metadata)
                    collection.add(
                        documents=[paragraph],
                        ids=[document["id"]],
                        embeddings=[document["embedding"]],
                        metadatas=[{"url": document["url"], "text": document["text"]}]
                    )

                print(f"[{idx+1}] Embedded {len(paragraphs)} chunks from {article_url}")
            else:
                print(f"[{idx+1}] Content extraction returned None.")
        else:
            print(f"[{idx+1}] Failed to fetch: {article_url}")
else:
    print(f"NewsAPI request failed: {response.status_code}")

print("Embeddings generation process done!")

