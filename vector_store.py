# vector_store.py
# Using FAISS instead of ChromaDB - lighter and works on Python 3.14!
# FAISS = Facebook AI Similarity Search - very popular in production RAG systems

import os
import json
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from config import GEMINI_API_KEY, TOP_K_RESULTS

genai.configure(api_key=GEMINI_API_KEY)

# Where we save our index files on disk
INDEX_PATH = "./faiss_index/index.faiss"
METADATA_PATH = "./faiss_index/metadata.json"
TEXTS_PATH = "./faiss_index/texts.pkl"


def get_embedding(text):
    """Converts text into a vector using Gemini embedding API."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]


def get_query_embedding(text):
    """Converts a question into a vector."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return result["embedding"]


def store_documents(chunks):
    """Stores document chunks as vectors in FAISS."""
    os.makedirs("./faiss_index", exist_ok=True)

    texts = []
    embeddings = []
    metadatas = []

    print(f"Processing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        print(f"Embedding chunk {i+1}/{len(chunks)}...")
        embedding = get_embedding(text)
        texts.append(text)
        embeddings.append(embedding)
        metadatas.append({
            "source": chunk.metadata.get("source", "unknown"),
            "page": str(chunk.metadata.get("page", "unknown")),
            "chunk_index": str(i)
        })

    # Convert to numpy array (FAISS needs this format)
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # Create FAISS index
    dimension = len(embeddings[0])  # Size of each vector (768 for Gemini)
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine similarity)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)

    # Save everything to disk
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadatas, f)

    print(f"Successfully stored {len(chunks)} chunks! ✓")
    return len(chunks)


def search_similar_chunks(query, n_results=TOP_K_RESULTS):
    """Finds the most relevant chunks for a question."""
    if not os.path.exists(INDEX_PATH):
        return []

    # Load saved index
    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)
    with open(METADATA_PATH, "r") as f:
        metadatas = json.load(f)

    # Convert question to vector
    query_embedding = np.array([get_query_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    # Search for most similar chunks
    n = min(n_results, len(texts))
    scores, indices = index.search(query_embedding, n)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "text": texts[idx],
                "metadata": metadatas[idx],
                "similarity_score": float(score)
            })

    return results


def clear_collection():
    """Deletes all stored vectors."""
    import shutil
    if os.path.exists("./faiss_index"):
        shutil.rmtree("./faiss_index")
    print("Collection cleared! ✓")


def get_collection_count():
    """Returns how many chunks are stored."""
    if not os.path.exists(TEXTS_PATH):
        return 0
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)
    return len(texts)