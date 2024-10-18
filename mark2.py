from openai import OpenAI
import os
import faiss
import numpy as np
import json

# Initialize OpenAI client
client = OpenAI()

# Load articles from folder
def load_articles_from_folder(folder_path):
    articles = []
    file_names = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                articles.append(file.read())
                file_names.append(filename)
    
    return articles, file_names

# Function to get embeddings from OpenAI using new method
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")  # Clean up newlines
    embedding_response = client.embeddings.create(input=[text], model=model)
    return embedding_response.data[0].embedding  # Extract the embedding

# Function to save embeddings to a JSON file
def save_embeddings_to_file(embeddings, file_names, file_path="embeddings.json"):
    data_to_store = [{"file_name": file_name, "embedding": embedding} for file_name, embedding in zip(file_names, embeddings)]
    
    with open(file_path, 'w') as f:
        json.dump(data_to_store, f)

# Function to load embeddings from a JSON file
def load_embeddings_from_file(file_path="embeddings.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    file_names = [item["file_name"] for item in data]
    embeddings = [item["embedding"] for item in data]
    
    return file_names, np.array(embeddings)

# Load articles from the 'data' folder
articles, file_names = load_articles_from_folder("data_tax")

# Generate embeddings for each article (if not already stored)
if not os.path.exists("embeddings.json"):
    embeddings = [get_embedding(article) for article in articles]
    save_embeddings_to_file(embeddings, file_names)
    print("Embeddings generated and saved.")
else:
    print("Embeddings file already exists. Skipping embedding generation.")

# Load embeddings from file
file_names, embeddings_array = load_embeddings_from_file()

# Build FAISS index
dimension = len(embeddings_array[0])  # Embedding size
faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance metric
faiss_index.add(embeddings_array)  # Add all article embeddings to FAISS index

# Function to search for similar articles based on a query
def search_similar_articles(query, k=5, model="text-embedding-3-small"):
    # Get embedding for the query from OpenAI
    query_embedding = get_embedding(query, model=model)
    
    # Convert query embedding to NumPy array
    query_embedding_np = np.array([query_embedding])
    
    # Search FAISS index for the top-k similar articles
    distances, indices = faiss_index.search(query_embedding_np, k)
    
    # Return file names and similarity scores (distances)
    results = [(file_names[i], distances[0][rank]) for rank, i in enumerate(indices[0])]
    
    # Sort the results by ranking (smaller distance means more similarity)
    results_sorted = sorted(results, key=lambda x: x[1])
    
    # Display results with ranking
    print("Results ranked by similarity (lower distance = higher similarity):")
    for rank, (article, score) in enumerate(results_sorted, 1):
        print(f"{rank}. {article} (Similarity Score: {score})")
    
    return results_sorted

# Example search query
query = "What credits and deductions are there for the new york state tax system?"
similar_articles_with_scores = search_similar_articles(query, k=3)
