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

# --- Load both FAISS indexes and metadata ---
print("Loading both indexes...")

indexes = {
    "volume1": {
        "index": faiss.read_index("faiss_partb/index.faiss"),
        "metadata": pickle.load(open("faiss_partb/part_metadata.pkl", "rb")),
        "chunks": json.load(open("PartB_chunks.json", "r"))
    },
    "volume2": {
        "index": faiss.read_index("faiss_partb_volume2/index.faiss"),
        "metadata": pickle.load(open("faiss_partb_volume2/part_metadata.pkl", "rb")),
        "chunks": json.load(open("PartB_volume2_chunks.json", "r"))
    }
}
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Both indexes loaded.")

# --- Core functions ---
def query_index(question, building_type="volume1", top_k=5):
    q_embedding = model.encode([question])
    index_data = indexes[building_type]
    D, I = index_data["index"].search(np.array(q_embedding), top_k)
    results = []
    for i in I[0]:
        entry = index_data["metadata"][i]
        results.append({
            "section": entry["section"],
            "heading": entry["heading"],
            "text": index_data["chunks"][i]["text"]
        })
    return results

def generate_answer(question, context_chunks, conversation_history=None):
    context = "\n\n---\n\n".join(
        [f"Section {c['section']} - {c['heading']}\n{c['text']}" for c in context_chunks]
    )

    system_prompt = """
You are an assistant trained to answer questions using UK Building Regulations, specifically Approved Document B (Fire Safety).

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
        building_type = data_in.get("building_type", "volume1")

        if not question:
            return jsonify({"error": "No question provided"}), 400
        if building_type not in indexes:
            return jsonify({"error": f"Unknown building type: {building_type}"}), 400

        top_chunks = query_index(question, building_type=building_type)
        answer = generate_answer(question, top_chunks, conversation_history=history)

        is_clarify = answer.strip().startswith("[[CLARIFY]]")
        return jsonify({"answer": answer, "clarify": is_clarify})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
