# app.py
# This is the main file that runs our web application
# Streamlit turns this Python code into a beautiful web interface
# To run it, type: streamlit run app.py in your terminal

import streamlit as st
from document_processor import load_and_split_document, get_document_stats
from vector_store import store_documents, clear_collection, get_collection_count
from rag_engine import get_answer, test_gemini_connection


# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# This must be the first Streamlit command in the file
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",  # Use full width of the screen
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────
# CUSTOM CSS STYLING
# We add a little CSS to make our app look more professional
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .answer-box {
    background-color: #1e1e2e;
    border-left: 4px solid #667eea;
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1rem 0;
    color: #ffffff;
            
    }
    .source-box {
        background-color: #1e1e2e;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# Session state lets us remember things between interactions
# Think of it like short-term memory for our app
# Without this, every button click would reset everything!
# ─────────────────────────────────────────────
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False  # Has a document been uploaded?

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of all questions and answers

if "document_name" not in st.session_state:
    st.session_state.document_name = ""  # Name of the uploaded document


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-header">🧠 DocuMind AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload any document and have an intelligent conversation with it</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# The sidebar is where users upload documents and see app info
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Upload")
    st.write("Upload a PDF or TXT file to get started")

    # File uploader widget
    # This creates a drag-and-drop upload area
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],  # Only allow PDF and TXT files
        help="Supported formats: PDF, TXT"
    )

    # Process button - only show if a file is uploaded
    if uploaded_file is not None:
        st.info(f"📄 **{uploaded_file.name}** selected")

        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            # Show a spinner while processing
            with st.spinner("Processing your document... This may take a moment!"):
                try:
                    # Step 1: Clear any previously stored documents
                    clear_collection()

                    # Step 2: Load and split the document into chunks
                    st.write("📖 Reading document...")
                    chunks = load_and_split_document(uploaded_file)

                    # Step 3: Store chunks in ChromaDB
                    st.write("🧠 Learning from document...")
                    store_documents(chunks)

                    # Step 4: Get stats about the processed document
                    stats = get_document_stats(chunks)

                    # Update session state
                    st.session_state.document_loaded = True
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.chat_history = []  # Clear old chat

                    # Show success message with stats
                    st.success("✅ Document processed successfully!")
                    st.write(f"📊 **Stats:**")
                    st.write(f"- Chunks created: **{stats['total_chunks']}**")
                    st.write(f"- Total characters: **{stats['total_characters']:,}**")
                    st.write(f"- Avg chunk size: **{stats['avg_chunk_size']}** chars")

                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")

    # Divider
    st.divider()

    # Show current status
    st.header("📊 Status")
    if st.session_state.document_loaded:
        st.success(f"✅ Document loaded")
        st.write(f"📄 **{st.session_state.document_name}**")
        chunk_count = get_collection_count()
        st.write(f"🧩 **{chunk_count}** chunks in memory")

        # Button to clear the document
        if st.button("🗑️ Clear Document", use_container_width=True):
            clear_collection()
            st.session_state.document_loaded = False
            st.session_state.chat_history = []
            st.session_state.document_name = ""
            st.rerun()
    else:
        st.warning("⚠️ No document loaded")

    st.divider()

    # Gemini connection status
    st.header("🔌 Connection")
    if st.button("Test Gemini Connection", use_container_width=True):
        with st.spinner("Testing..."):
            success, message = test_gemini_connection()
            if success:
                st.success(f"✅ Connected!\n{message}")
            else:
                st.error(f"❌ Failed: {message}")

    st.divider()

    # About section
    st.header("ℹ️ About")
    st.write("""
    **DocuMind AI** uses RAG (Retrieval-Augmented Generation) to answer questions from your documents.
    
    **How it works:**
    1. 📄 Upload your document
    2. 🧩 It gets split into chunks
    3. 🧠 Chunks are stored with meaning
    4. ❓ You ask a question
    5. 🔍 Relevant chunks are found
    6. 🤖 Gemini answers using those chunks
    """)


# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────
if not st.session_state.document_loaded:
    # Show welcome screen if no document is loaded
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📤 Step 1
        **Upload your document**
        
        Upload any PDF or TXT file using the sidebar on the left
        """)

    with col2:
        st.markdown("""
        ### ⚙️ Step 2
        **Process it**
        
        Click "Process Document" and wait for DocuMind to learn from it
        """)

    with col3:
        st.markdown("""
        ### 💬 Step 3
        **Start chatting!**
        
        Ask any question about your document and get instant AI-powered answers
        """)

    st.markdown("---")
    st.info("👈 Start by uploading a document in the sidebar!")

else:
    # Show chat interface if document is loaded
    st.markdown(f"### 💬 Chat with **{st.session_state.document_name}**")
    st.write("Ask me anything about your document!")

    # Display chat history
    # This loops through all previous questions and answers and displays them
    for chat in st.session_state.chat_history:
        # Display user question
        with st.chat_message("user"):
            st.write(chat["question"])

        # Display AI answer
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(f'<div class="answer-box">{chat["answer"]}</div>',
                       unsafe_allow_html=True)

            # Show sources in an expandable section
            if chat["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(chat["sources"]):
                        st.markdown(f"""
                        <div class="source-box">
                        <b>Source {i+1}</b> | Page: {source['page']} | 
                        Relevance: {source['similarity']}%<br>
                        <small>{source['preview']}</small>
                        </div>
                        """, unsafe_allow_html=True)

    # Question input at the bottom
    # st.chat_input creates a chat-style input box at the bottom of the page
    question = st.chat_input("Ask a question about your document...")

    if question:
        # Show the user's question immediately
        with st.chat_message("user"):
            st.write(question)

        # Get and show the answer
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("🔍 Searching document and generating answer..."):
                result = get_answer(question)

            # Display the answer
            st.markdown(f'<div class="answer-box">{result["answer"]}</div>',
                       unsafe_allow_html=True)

            # Display sources
            if result["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(result["sources"]):
                        st.markdown(f"""
                        <div class="source-box">
                        <b>Source {i+1}</b> | Page: {source['page']} | 
                        Relevance: {source['similarity']}%<br>
                        <small>{source['preview']}</small>
                        </div>
                        """, unsafe_allow_html=True)

        # Save this conversation to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        })