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
model = SentenceTransformer("all-MiniLM-L6-v2")

# Volume 1
index_v1 = faiss.read_index("partb_index.faiss")
with open("partb_metadata.pkl", "rb") as f:
    metadata_v1 = pickle.load(f)
with open("PartB_chunks.json", "r") as f:
    data_v1 = json.load(f)

# Volume 2
index_v2 = faiss.read_index("faiss_partb_volume2/index.faiss")
with open("faiss_partb_volume2/index.pkl", "rb") as f:
    metadata_v2 = pickle.load(f)
with open("partb_volume2_chunks.json", "r") as f:
    data_v2 = json.load(f)

print("Data loaded.")

# --- Core functions ---
def query_index(question, volume="volume1", top_k=5):
    q_embedding = model.encode([question])
    
    if volume == "volume2":
        D, I = index_v2.search(np.array(q_embedding), top_k)
        results = []
        for i in I[0]:
            entry = metadata_v2[i]
            results.append({
                "section": entry.get("section", ""),
                "heading": entry.get("heading", ""),
                "text": data_v2[i]["content"]
            })
        return results
    else:
        D, I = index_v1.search(np.array(q_embedding), top_k)
        results = []
        for i in I[0]:
            entry = metadata_v1[i]
            results.append({
                "section": entry.get("section", ""),
                "heading": entry.get("heading", ""),
                "text": data_v1[i]["text"]
            })
        return results

def generate_answer(question, context_chunks, conversation_history=None, volume="volume1"):
    # Update system prompt depending on document
    if volume == "volume2":
        doc_type = "Approved Document B Volume 2 (Fire Safety – Buildings other than dwellinghouses)"
    else:
        doc_type = "Approved Document B Volume 1 (Fire Safety – Dwellings)"

    context = "\n\n---\n\n".join(
        [f"Section {c['section']} - {c['heading']}\n{c['text']}" for c in context_chunks]
    )

    system_prompt = f"""
You are an assistant trained to answer questions using UK Building Regulations, specifically {doc_type}.

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
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        data_in = request.get_json()
        question = data_in.get("question")
        history = data_in.get("history", [])
        building_type = data_in.get("building_type", "volume1").lower()

        if not question:
            return jsonify({"error": "No question provided"}), 400

        chunks = query_index(question, volume=building_type)
        answer = generate_answer(question, chunks, conversation_history=history, volume=building_type)

        is_clarify = answer.strip().startswith("[[CLARIFY]]")
        return jsonify({"answer": answer, "clarify": is_clarify})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
