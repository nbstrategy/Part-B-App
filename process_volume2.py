import json
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Step 1: Load and clean raw text ---
with open("partb_volume2_raw.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Basic cleanup (can be extended if needed)
cleaned_text = raw_text.replace('\r', '').strip()

# --- Step 2: Chunking with metadata scaffold ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_text(cleaned_text)

chunk_data = []
metadata = []

for i, chunk in enumerate(chunks):
    chunk_data.append({"text": chunk})
    metadata.append({
        "section": "Unknown",  # You can fill these in manually later if desired
        "heading": "Unknown"
    })

# Create output directory if it doesn't exist
os.makedirs("faiss_partb_volume2", exist_ok=True)

with open("faiss_partb_volume2/PartB_volume2_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, indent=2)

with open("faiss_partb_volume2/partb_volume2_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# --- Step 3: Create FAISS index ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([chunk["text"] for chunk in chunk_data])
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "faiss_partb_volume2/partb_volume2_index.faiss")

print(f"✅ Done. {len(chunks)} chunks processed and indexed.")
