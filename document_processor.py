# document_processor.py
# This file is responsible for:
# 1. Reading PDF or text files uploaded by the user
# 2. Extracting the raw text from them
# 3. Splitting that text into smaller chunks for processing

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import tempfile
import os
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_and_split_document(uploaded_file):
    """
    Takes an uploaded file from Streamlit,
    saves it temporarily, reads it, and splits it into chunks.
    """

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        elif file_extension == ".txt":
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                text = f.read()
            documents = [Document(page_content=text, metadata={"source": uploaded_file.name})]
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)
        return chunks

    finally:
        os.unlink(tmp_file_path)


def get_document_stats(chunks):
    """
    Returns simple statistics about the processed document.
    """
    total_chunks = len(chunks)
    total_characters = sum(len(chunk.page_content) for chunk in chunks)
    avg_chunk_size = total_characters // total_chunks if total_chunks > 0 else 0

    return {
        "total_chunks": total_chunks,
        "total_characters": total_characters,
        "avg_chunk_size": avg_chunk_size
    }