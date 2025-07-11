from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
import faiss
import pickle
import json
import numpy as np
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# --- Config ---
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# --- App setup ---
app = Flask(__name__)

# --- Load data ---
print("Loading data...")
index = faiss.read_index("partb_index.faiss")
with open("partb_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
with open("PartB_chunks.json", "r") as f:
    data = json.load(f)
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Data loaded.")

# --- Core functions ---
def query_index(question, top_k=5):
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), top_k)
    results = []
    for i in I[0]:
        entry = metadata[i]
        results.append({
            "section": entry["section"],
            "heading": entry["heading"],
            "text": data[i]["text"]
        })
    return results

def generate_answer(question, context_chunks, conversation_history=None):
    context = "\n\n---\n\n".join(
        [f"Section {c['section']} - {c['heading']}\n{c['text']}" for c in context_chunks]
    )

    system_prompt = """
You are an assistant trained to answer questions using UK Building Regulations, specifically Approved Document B Volume 1 (Fire Safety – Dwellings).

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
    
    # Add conversation history if present
    if conversation_history:
        messages.extend(conversation_history)

    # Add current user input
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        data_in = request.get_json()
        question = data_in.get("question")
        history = data_in.get("history", [])

        if not question:
            return jsonify({"error": "No question provided"}), 400

        top_chunks = query_index(question)
        answer = generate_answer(question, top_chunks, conversation_history=history)

        is_clarify = answer.strip().startswith("[[CLARIFY]]")
        return jsonify({"answer": answer, "clarify": is_clarify})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
