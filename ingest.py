"""
ingest.py — Embed Flickr30k subset into a SINGLE ChromaDB collection.
Both images (via URI) and captions (via text) are embedded by OpenCLIP
into the same shared vector space.

Run once. Re-running skips if collection is already populated.

Expected layout:
    data/
    ├── images/      ← .jpg files
    └── captions.txt ← image_name, comment_number, comment
"""

import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from tqdm import tqdm
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
IMAGES_DIR    = os.path.join(DATA_DIR, "images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.txt")
CHROMA_PATH   = "./chroma_db"
COLLECTION    = "flickr30k_subset"
BATCH_SIZE    = 50
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── 1. Load captions ──────────────────────────────────────────────────────
    print("Loading captions...")
    df = pd.read_csv(CAPTIONS_FILE)
    df.columns = df.columns.str.strip()
    unique_images = df["image_name"].unique().tolist()
    print(f"  {len(unique_images)} unique images | {len(df)} total captions")

    # ── 2. Connect to ChromaDB ────────────────────────────────────────────────
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # OpenCLIP maps BOTH text and images into the same 512-d vector space
    ef = OpenCLIPEmbeddingFunction()
    il = ImageLoader()

    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        col = client.get_collection(name=COLLECTION, embedding_function=ef, data_loader=il)
        if col.count() > 0:
            print(f"  '{COLLECTION}' already has {col.count()} items — skipping.")
            print("  Delete ./chroma_db to re-ingest.")
            return
    else:
        col = client.create_collection(
            name=COLLECTION,
            embedding_function=ef,
            data_loader=il,
            # cosine distance so image<->text similarity is meaningful
            metadata={"hnsw:space": "cosine"},
        )

    # ── 3. Add images ─────────────────────────────────────────────────────────
    # ChromaDB calls OpenCLIP's encode_image() on each URI automatically.
    print("\nIngesting images into shared CLIP space...")
    img_ids, img_uris, img_metas = [], [], []

    for img_name in unique_images:
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        img_ids.append(f"img__{img_name}")
        img_uris.append(img_path)
        img_metas.append({"type": "image", "image_name": img_name})

    for i in tqdm(range(0, len(img_ids), BATCH_SIZE), desc="Image batches"):
        col.add(
            ids=img_ids[i : i + BATCH_SIZE],
            uris=img_uris[i : i + BATCH_SIZE],
            metadatas=img_metas[i : i + BATCH_SIZE],
        )

    # ── 4. Add captions ───────────────────────────────────────────────────────
    # ChromaDB calls OpenCLIP's encode_text() on each document automatically.
    # Both embeddings land in the same 512-d CLIP space => cross-modal retrieval.
    print("\nIngesting captions into shared CLIP space...")
    txt_ids, txt_docs, txt_metas = [], [], []

    for _, row in df.iterrows():
        img_name  = row["image_name"]
        comment   = str(row["comment"]).strip()
        cap_num   = int(row["comment_number"])
        txt_ids.append(f"cap__{img_name}__{cap_num}")
        txt_docs.append(comment)
        txt_metas.append({
            "type"          : "text",
            "image_name"    : img_name,
            "comment_number": cap_num,
        })

    for i in tqdm(range(0, len(txt_ids), BATCH_SIZE), desc="Caption batches"):
        col.add(
            ids=txt_ids[i : i + BATCH_SIZE],
            documents=txt_docs[i : i + BATCH_SIZE],
            metadatas=txt_metas[i : i + BATCH_SIZE],
        )

    print(f"\nDone! Total items: {col.count()}")
    print(f"   Images  : {len(img_ids)}")
    print(f"   Captions: {len(txt_ids)}")
    print(f"   All embedded in the same CLIP vector space (cosine distance).")


if __name__ == "__main__":
    main()