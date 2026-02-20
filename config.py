# config.py
# This file loads our secret API key and stores all settings in one place
# Think of it as the "settings page" of our app

import os
from dotenv import load_dotenv

# load_dotenv() reads our .env file and makes the values available to Python
load_dotenv()

# Grab the Gemini API key from the .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# The Gemini model we want to use
GEMINI_MODEL = "gemini-2.0-flash"  # This is free and very capable

# Embedding model - this runs locally on your machine, no API needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, fast, and accurate

# ChromaDB settings - where we store our document chunks
CHROMA_DB_PATH = "./chroma_db"  # A folder that will be created automatically
COLLECTION_NAME = "documind_collection"

# How we split documents into chunks
# Chunk size = how many characters per chunk
CHUNK_SIZE = 1000
# Chunk overlap = how many characters overlap between chunks
# (so we don't lose meaning at the edges of chunks)
CHUNK_OVERLAP = 200

# How many relevant chunks to retrieve when answering a question
TOP_K_RESULTS = 5