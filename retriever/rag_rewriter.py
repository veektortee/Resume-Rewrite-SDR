import faiss
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=api_key)
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index(index_path="embeddings/faiss_index.index"):
    return faiss.read_index(index_path)

def load_records(pkl_path="embeddings/faiss_index.pkl"):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)
    
def load_rewrite_rules(path="data/rewrite_prompts.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_prompt(raw_input, retrieved_texts):
    rules = load_rewrite_rules()
    instructions = "\n".join([f"- {rule}" for rule in rules])
    examples = "\n\n".join([f"--- 📄 Example {i+1} ---\n{txt}" for i, txt in enumerate(retrieved_texts)])
    return f"""You are a career coach who helps people land SDR (Sales Development Representative) roles by helping them rewrite their resumes.

Follow these rewriting instructions:
{instructions}

Reference the examples below to understand how to rewrite resumes effectively. Use the provided examples as a guide for structure, content, and style.
{examples}

--- Resume to Rewrite ---
{raw_input}

--- ✅ Rewritten SDR Resume ---
"""

def rewrite_resume(new_raw_resume, k=3):
    index = load_index()
    records = load_records()

    input_embedding = model.encode([new_raw_resume])
    _, I = index.search(input_embedding, k)

    retrieved_chunks = [records[i] for i in I[0]]
    prompt = build_prompt(new_raw_resume, retrieved_chunks)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": "You are a resume rewriting assistant that strictly follows formatting instructions to produce structured SDR resumes."},
                  {"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()
