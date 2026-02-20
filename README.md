# DocuMind AI

A RAG (Retrieval-Augmented Generation) chatbot that enables intelligent 
conversations with your documents. Upload any PDF or text file and ask 
questions — DocuMind retrieves the most relevant content and generates 
accurate, context-aware answers using Google Gemini AI.

## What is RAG?

RAG stands for Retrieval-Augmented Generation. Instead of relying on an 
AI model's general knowledge, RAG grounds responses in your specific 
documents. This means answers are accurate, verifiable, and sourced 
directly from your content.

The pipeline works as follows:
1. Your document is parsed and split into overlapping chunks of text
2. Each chunk is converted into a vector (numerical representation of meaning) using Gemini Embeddings
3. Vectors are stored in a FAISS index for fast similarity search
4. When you ask a question, it is also converted to a vector
5. The most semantically similar chunks are retrieved
6. Those chunks plus your question are sent to Gemini, which generates a grounded answer
7. The answer is returned along with the source chunks it was based on

## Features

- Upload PDF or TXT documents up to 200MB
- Intelligent semantic search powered by Google Gemini Embeddings
- AI-generated answers grounded strictly in your document content
- Source citations showing exactly where each answer came from
- Persistent chat history within a session
- Clean, intuitive web interface built with Streamlit
- Fully free to run using Google Gemini's free tier

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Gemini Embedding 001 |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Document Processing | LangChain + PyPDF |
| Frontend | Streamlit |
| Language | Python 3.x |

## Getting Started

### Prerequisites

- Python 3.9 or higher (note: Python 3.14 has some package compatibility 
  issues, Python 3.11 is recommended)
- A free Google Gemini API key from https://aistudio.google.com/app/apikey

### Installation

Clone the repository:
```bash
git clone https://github.com/winnerumukoro/documind-ai.git
cd documind-ai
```

Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

Install dependencies:
```bash
pip install streamlit google-generativeai faiss-cpu pypdf python-dotenv langchain-community langchain langchain-core
```

Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## Usage

1. Upload a PDF or TXT file using the sidebar
2. Click "Process Document" and wait for it to be indexed
3. Type any question about your document in the chat input
4. DocuMind will retrieve relevant sections and generate an answer
5. Expand "View Sources" to see exactly which parts of the document were used

## Project Structure
```
documind-ai/
├── app.py                  # Streamlit web interface
├── rag_engine.py           # Core RAG pipeline and Gemini integration
├── vector_store.py         # FAISS vector storage and retrieval
├── document_processor.py   # PDF/TXT parsing and text chunking
├── config.py               # Configuration and environment variables
├── requirements.txt        # Python dependencies
└── .env                    # API keys (never commit this file)
```

## Environment Variables

| Variable | Description |
|---|---|
| GEMINI_API_KEY | Your Google Gemini API key |

## Limitations

- The free Gemini API tier has daily request limits
- Very large documents may take longer to process due to embedding API calls
- Currently supports PDF and TXT formats only

## Future Improvements

- Support for DOCX, CSV, and web URLs
- Multi-document chat (ask across multiple files simultaneously)
- Conversation memory across sessions
- User authentication for personal document storage
- Docker deployment support

## Author

Built by [Winner Umukoro](https://github.com/winnerumukoro)

## License

MIT License — feel free to use, modify, and distribute this project.