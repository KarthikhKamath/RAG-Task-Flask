from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import chromadb
from uuid import uuid4
import os


# Initialize Flask app
app = Flask(__name__)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

storage_path = os.path.join(os.getcwd(), "chroma_store")
if not os.path.exists(storage_path):
    os.makedirs(storage_path)

# Initialize Chroma client with new API
client = chromadb.PersistentClient(path=storage_path)

def list_collections(client):
    try:
        collections = client.list_collections()
        print(f"Collections found: {collections}")  # Debugging line
        return [{"name": c.name, "metadata": c.metadata} for c in collections]
    except Exception as e:
        print(f"Error while listing collections: {str(e)}")
        return []


@app.route('/list-collections', methods=['GET'])
def list_collections_endpoint():
    try:
        collections = list_collections(client)
        if collections:
            return jsonify({
                "collections": collections
            }), 200
        else:
            return jsonify({"message": "No collections found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to query the vector DB
def query_vector_db(query, collection, model, top_k=5):
    query_embedding = model.encode(query, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    relevant_paragraphs = []
    for idx in range(len(results['documents'][0])): 
        relevant_paragraphs.append({
            'rank': idx + 1,
            'text': results['documents'][0][idx],
            'metadata': results['metadatas'][0][idx]
        })
    return relevant_paragraphs

@app.route('/query', methods=['POST'])
def query():
    try:
        user_query = request.json.get('query')
        numbers = request.json.get('n_results')
        collection_name = request.json.get('collection', 'news_articles') 

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        try:
            collection = client.get_collection(collection_name)
        except Exception as e:
            return jsonify({"error": f"Collection '{collection_name}' not found: {str(e)}"}), 404

        relevant_paragraphs = query_vector_db(user_query, collection, model, top_k=numbers)
        if not relevant_paragraphs:
            return jsonify({"message": "No relevant results found"}), 404

        return jsonify({
            "results": relevant_paragraphs
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
