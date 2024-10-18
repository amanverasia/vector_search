from openai import OpenAI
import os
import faiss
import numpy as np

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

# Load articles from the 'data' folder
articles, file_names = load_articles_from_folder("data_tax")

# Generate embeddings for each article
embeddings = [get_embedding(article) for article in articles]

# Convert embeddings into NumPy array
embeddings_array = np.array(embeddings)

# Build FAISS index
dimension = len(embeddings[0])  # Embedding size
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
query = "What is the current sales tax rate in New York State?"
similar_articles = search_similar_articles(query, k=3)

# Output similar articles
print("Similar articles:", similar_articles)
