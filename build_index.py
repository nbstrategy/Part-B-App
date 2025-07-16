import os
import faiss
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, max_tokens=200):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    sentences = text.split("\n")
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if sentence.strip() == "":
            continue
        current_chunk.append(sentence.strip())
        if len(enc.encode(" ".join(current_chunk))) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def extract_sections_for_chunks(chunks):
    import re
    current_section = "Unknown"
    current_heading = "Unknown"
    metadata = []

    for chunk in chunks:
        section_match = re.search(r"Section\s+(\d+[A-Z]?)", chunk)
        heading_match = re.search(r"Section\s+\d+[A-Z]?\s*–\s*(.+)", chunk)

        if section_match:
            current_section = f"Section {section_match.group(1)}"

        if heading_match:
            current_heading = heading_match.group(1).strip()

        metadata.append({
            "section": current_section,
            "heading": current_heading
        })

    return metadata


def build_faiss_index(chunks, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    metadata = extract_sections_for_chunks(chunks)
    embeddings = [model.encode(chunk) for chunk in chunks]
    chunk_dicts = [{"text": chunk} for chunk in chunks]

    embeddings_np = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    faiss.write_index(index, os.path.join(out_dir, "partb_volume2_index.faiss"))

    with open(os.path.join(out_dir, "partb_volume2_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    with open(os.path.join(out_dir, "PartB_volume2_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, indent=2)

    print("Indexing complete:", len(chunk_dicts), "chunks")


# --- RUN ---
text = load_text("partb_volume2_raw.txt")
chunks = split_text(text)
build_faiss_index(chunks, "faiss_partb_volume2")

