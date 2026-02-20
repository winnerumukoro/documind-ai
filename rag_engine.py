# rag_engine.py
# This is the core of our RAG system - it connects everything together
# It takes a user's question, finds relevant context, and asks Gemini to answer

import google.generativeai as genai
from vector_store import search_similar_chunks
from config import GEMINI_API_KEY, GEMINI_MODEL


# Configure Gemini with our API key
# This is like "logging in" to the Gemini service
genai.configure(api_key=GEMINI_API_KEY)


def build_prompt(question, context_chunks):
    """
    Builds the prompt we send to Gemini.
    
    A prompt is basically the instruction + context + question we give to the AI.
    We're telling Gemini:
    - Here is some text from a document
    - Only use THIS text to answer
    - Here is the user's question
    
    This is important because we don't want Gemini making things up!
    We want answers ONLY from the uploaded document.
    """

    # Combine all the relevant chunks into one block of context
    context = "\n\n---\n\n".join([chunk["text"] for chunk in context_chunks])

    # This is our prompt template
    # We're giving Gemini clear instructions on how to behave
    prompt = f"""You are a helpful AI assistant called DocuMind. 
Your job is to answer questions based ONLY on the document context provided below.

IMPORTANT RULES:
- Only use information from the context below to answer
- If the answer is not in the context, say "I couldn't find that information in the uploaded document"
- Be clear, concise, and helpful
- If relevant, mention which part of the document your answer comes from

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

YOUR ANSWER:"""

    return prompt


def get_answer(question):
    """
    Main function that:
    1. Takes the user's question
    2. Finds relevant chunks from our vector store
    3. Builds a prompt with those chunks
    4. Sends it to Gemini
    5. Returns Gemini's answer

    This is the full RAG pipeline in one function!
    """

    # Step 1: Find the most relevant chunks for this question
    # This searches ChromaDB for chunks that are semantically similar to the question
    relevant_chunks = search_similar_chunks(question)

    # If no chunks found, the document probably hasn't been uploaded yet
    if not relevant_chunks:
        return {
            "answer": "No document found! Please upload a document first.",
            "sources": [],
            "relevant_chunks": []
        }

    # Step 2: Build our prompt using the question + relevant chunks
    prompt = build_prompt(question, relevant_chunks)

    # Step 3: Send the prompt to Gemini and get a response
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Generate the response
        # Think of this as "asking Gemini the question with context"
        response = model.generate_content(prompt)

        # Extract the text from the response
        answer = response.text

        # Step 4: Prepare source information to show the user
        # This tells the user WHERE in the document the answer came from
        sources = []
        for chunk in relevant_chunks:
            source_info = {
                "page": chunk["metadata"].get("page", "unknown"),
                "source": chunk["metadata"].get("source", "unknown"),
                "similarity": round(chunk["similarity_score"] * 100, 1),  # As percentage
                "preview": chunk["text"][:200] + "..."  # First 200 chars as preview
            }
            sources.append(source_info)

        return {
            "answer": answer,
            "sources": sources,
            "relevant_chunks": relevant_chunks
        }

    except Exception as e:
        # If something goes wrong with Gemini, return a friendly error
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": [],
            "relevant_chunks": []
        }


def test_gemini_connection():
    """
    Simple function to test if our Gemini API key works.
    We call this when the app starts to make sure everything is connected.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content("Say 'DocuMind is ready!' in exactly those words.")
        return True, response.text
    except Exception as e:
        return False, str(e)