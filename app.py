# Part B Fire Safety Assistant — app.py
# Version: 1.3.0
# Features: FAISS-based search, volume switch, clarification loop, basic auth (Flask-HTTPAuth)

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- Config ---
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Auth setup ---
auth = HTTPBasicAuth()
users = {
    "Admin": generate_password_hash("PBA888"),
    "NNA": generate_password_hash("NNAtest1"),
    "FloydSlaski": generate_password_hash("FStest849"),
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

# --- Load indexes ---
def load_index_set(path):
    index = faiss.read_index(os.path.join(path, [f for f in os.listdir(path) if f.endswith(".faiss")][0]))
    with open(os.path.join(path, [f for f in os.listdir(path) if f.endswith("metadata.pkl")][0]), "rb") as f:
        metadata = pickle.load(f)
    with open(os.path.join(path, [f for f in os.listdir(path) if f.endswith(".json")][0]), "r") as f:
        chunks = json.load(f)
    return index, metadata, chunks

volume1_index, volume1_metadata, volume1_chunks = load_index_set("faiss_partb")
volume2_index, volume2_metadata, volume2_chunks = load_index_set("faiss_partb_volume2")

# --- App setup ---
app = Flask(__name__)

# --- Core functions ---
def query_index(question, index, metadata, chunks, top_k=5):
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), top_k)
    results = []
    for i in I[0]:
        entry = metadata[i]
        results.append({
            "section": entry["section"],
            "heading": entry["heading"],
            "text": chunks[i]["text"]
        })
    return results

def generate_answer(question, context_chunks, conversation_history=None):
    context = "\n\n---\n\n".join(
        [f"Section {c['section']} – {c['heading']}\n{c['text']}" for c in context_chunks]
    )

    system_prompt = """
You are an assistant trained to answer questions using UK Building Regulations, specifically Approved Document B Volumes 1 and 2 (Fire Safety – Dwellings and Other Buildings).

Use the provided context ONLY.

If the question is even partially ambiguous or unclear, or could be interpreted in more than one way, instead of answering, respond with a clarifying question using the following format exactly:

[[CLARIFY]]
Your clarifying question here.

Otherwise, respond in this structured format:

1. **Answer**: Provide a concise, authoritative answer.
2. **Confidence Level**: Rate as High / Medium / Low and explain your reasoning based on the clarity and specificity of the guidance.
3. **Ambiguity or Discretion Flags**: List any phrases that suggest professional judgement, local authority approval, or unclear application.
4. **Constraints or Conditions**: Highlight any assumptions or conditions that limit the answer’s applicability.
5. **Further Sections to Consider**: List any relevant related sections or clauses not used in the main answer.
6. **Source References**: Cite the sections and headings (e.g., "Section 2.7 – Protected Stairways") that support your answer.
"""

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content

# --- Routes ---
@app.route("/")
@auth.login_required
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
@auth.login_required
def query():
    try:
        data_in = request.get_json()
        question = data_in.get("question")
        history = data_in.get("history", [])
        building_type = data_in.get("building_type", "volume1")

        print("BUILDING TYPE SELECTED:", building_type)

        if building_type == "volume2":
            idx, meta, chunks = volume2_index, volume2_metadata, volume2_chunks
        else:
            idx, meta, chunks = volume1_index, volume1_metadata, volume1_chunks

        top_chunks = query_index(question, idx, meta, chunks)
        print("TOP CHUNKS RETURNED:", top_chunks)
        answer = generate_answer(question, top_chunks, conversation_history=history)
        is_clarify = answer.strip().startswith("[[CLARIFY]]")

        return jsonify({"answer": answer, "clarify": is_clarify})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
