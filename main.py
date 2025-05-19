# ✅ STEP 2: Import libraries
import fitz  # PyMuPDF
import numpy as np
import faiss
import gradio as gr
from openai import OpenAI
from typing import List
from Key import keys  # Ensure you have a Key.py file with your OpenAI API key

# ✅ STEP 3: Set up OpenAI client
client = OpenAI(api_key=keys)  # Replace with your actual OpenAI API key

# ✅ STEP 4: Helper functions

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split text into chunks
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Get embedding using OpenAI v1.x
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 1536

# Build FAISS index
def build_faiss_index(chunks):
    dim = len(get_embedding("test"))
    index = faiss.IndexFlatL2(dim)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

# Search top K relevant chunks
def search_index(question, index, chunk_texts, k=3):
    q_embedding = np.array([get_embedding(question)]).astype("float32")
    distances, indices = index.search(q_embedding, k)
    return [chunk_texts[i] for i in indices[0]]

# Generate answer using OpenAI Chat API (v1.x)
def generate_answer(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {str(e)}"

# Main pipeline function
def qa_pipeline(pdf_file, question):
    text = extract_text_from_pdf(pdf_file.name)
    chunks = split_text(text)
    index, _ = build_faiss_index(chunks)
    relevant_chunks = search_index(question, index, chunks)
    context = " ".join(relevant_chunks)
    return generate_answer(question, context)

# ✅ STEP 5: Gradio interface
iface = gr.Interface(
    fn=qa_pipeline,
    inputs=[
        gr.File(label="Upload your PDF"),
        gr.Textbox(label="Ask a Question")
    ],
    outputs="text",
    title="📄 AI-Powered PDF Q&A",
    description="Upload a PDF document and ask any question. Uses OpenAI, FAISS, and PyMuPDF."
)

iface.launch(share=True)