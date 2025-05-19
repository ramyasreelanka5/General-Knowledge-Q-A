# ✅ STEP 1: Import libraries
import fitz  # PyMuPDF
import numpy as np
import faiss
import gradio as gr
from openai import OpenAI
from typing import List
from Key import keys  # Loads the API key from Key.py

# ✅ STEP 2: Setup OpenAI client
client = OpenAI(api_key=keys)

# ✅ STEP 3: Cache to avoid redundant processing
pdf_cache = {}

# Similarity threshold for FAISS distances (lower means more similar)
SIMILARITY_THRESHOLD = 0.5  # Adjust this based on your data

# ✅ STEP 4: Helper functions

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split into chunks
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Get embedding using OpenAI v1
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 1536

# Build FAISS index
def build_faiss_index(chunks):
    dim = len(get_embedding("sample text"))
    index = faiss.IndexFlatL2(dim)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

# Search top K relevant chunks applying similarity threshold
def search_index(question, index, chunk_texts, k=3):
    q_embedding = np.array([get_embedding(question)]).astype("float32")
    distances, indices = index.search(q_embedding, k)
    
    filtered_chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < SIMILARITY_THRESHOLD:
            filtered_chunks.append(chunk_texts[idx])
    return filtered_chunks

# Generate final answer using OpenAI Chat, strictly from context
def generate_answer(question, context):
    if not context.strip():
        return "Sorry, I couldn't find any relevant information in the document to answer your question."
    
    prompt = f"""You are an assistant that answers questions ONLY based on the provided context below.
If the answer is not contained in the context, respond with "I don't know."

Context: {context}

Question: {question}
Answer:"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {str(e)}"

# ✅ STEP 5: Main Q&A function
def qa_pipeline(pdf_file, question):
    file_path = pdf_file.name
    if file_path not in pdf_cache:
        text = extract_text_from_pdf(file_path)
        chunks = split_text(text)
        index, chunk_list = build_faiss_index(chunks)
        pdf_cache[file_path] = (index, chunk_list)
    else:
        index, chunk_list = pdf_cache[file_path]

    relevant_chunks = search_index(question, index, chunk_list)
    context = " ".join(relevant_chunks)
    return generate_answer(question, context)

# ✅ STEP 6: Gradio Interface
iface = gr.Interface(
    fn=qa_pipeline,
    inputs=[
        gr.File(label="📄 Upload PDF"),
        gr.Textbox(label="💬 Ask a Question")
    ],
    outputs="text",
    title="📘 AI-Powered PDF Q&A System",
    description="Upload a PDF and ask questions. Uses OpenAI + FAISS + PyMuPDF."
)

iface.launch(share=True)
