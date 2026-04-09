"""
Ingest Fashionpedia dataset into ChromaDB.

For each image, combines all annotation categories + attributes into a text
description, generates an OpenAI embedding, and stores it in ChromaDB.
"""

import json
import os
import time
from collections import defaultdict

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Paths
ANNOTATIONS_PATH = "data/annotations/instances_attributes_train2020.json"
IMAGES_DIR = "data/images/train"
CHROMA_DIR = "chroma_db"

# OpenAI
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request


def load_annotations(path):
    """Load and parse the Fashionpedia annotation JSON."""
    print(f"Loading annotations from {path}...")
    with open(path) as f:
        data = json.load(f)

    # Build lookup maps
    categories = {c["id"]: c["name"] for c in data["categories"]}
    attributes = {a["id"]: a["name"] for a in data["attributes"]}
    images = {img["id"]: img for img in data["images"]}

    # Group annotations by image_id
    image_annotations = defaultdict(list)
    for ann in data["annotations"]:
        image_annotations[ann["image_id"]].append(ann)

    print(f"  {len(images)} images, {len(data['annotations'])} annotations")
    print(f"  {len(categories)} categories, {len(attributes)} attributes")
    return images, image_annotations, categories, attributes


def build_image_descriptions(images, image_annotations, categories, attributes):
    """
    For each image, combine its annotation categories and attributes
    into a single text description for embedding.
    """
    descriptions = []
    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        image_path = os.path.join(IMAGES_DIR, file_name)

        # Skip images that don't exist locally
        if not os.path.exists(image_path):
            continue

        anns = image_annotations.get(image_id, [])
        if not anns:
            continue

        # Collect unique categories and attributes for this image
        cats = set()
        attrs = set()
        for ann in anns:
            cat_name = categories.get(ann["category_id"], "")
            if cat_name:
                cats.add(cat_name)
            for attr_id in ann.get("attribute_ids", []):
                attr_name = attributes.get(attr_id, "")
                if attr_name:
                    attrs.add(attr_name)

        # Build text description
        parts = []
        if cats:
            parts.append("Items: " + ", ".join(sorted(cats)))
        if attrs:
            parts.append("Attributes: " + ", ".join(sorted(attrs)))

        if parts:
            description = ". ".join(parts)
            descriptions.append({
                "id": str(image_id),
                "file_name": file_name,
                "description": description,
                "categories": list(cats),
                "attributes": list(attrs),
            })

    print(f"Built descriptions for {len(descriptions)} images")
    return descriptions


def embed_and_store(descriptions):
    """Generate embeddings and store in ChromaDB."""
    client = OpenAI()
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if re-running
    try:
        chroma.delete_collection("fashion")
    except Exception:
        pass

    collection = chroma.create_collection(
        name="fashion",
        metadata={"hnsw:space": "cosine"},
    )

    total = len(descriptions)
    print(f"Embedding and storing {total} images in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = descriptions[i : i + BATCH_SIZE]
        texts = [d["description"] for d in batch]

        # Generate embeddings
        response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        embeddings = [item.embedding for item in response.data]

        # Store in ChromaDB
        collection.add(
            ids=[d["id"] for d in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "file_name": d["file_name"],
                    "categories": json.dumps(d["categories"]),
                    "attributes": json.dumps(d["attributes"]),
                }
                for d in batch
            ],
        )

        processed = min(i + BATCH_SIZE, total)
        print(f"  [{processed}/{total}] ({processed * 100 // total}%)")

        # Rate limiting - be gentle with the API
        if i + BATCH_SIZE < total:
            time.sleep(0.5)

    print(f"\nDone! Stored {total} items in ChromaDB at ./{CHROMA_DIR}/")
    print(f"Collection '{collection.name}' has {collection.count()} entries")


def main():
    images, image_annotations, categories, attributes = load_annotations(ANNOTATIONS_PATH)
    descriptions = build_image_descriptions(images, image_annotations, categories, attributes)
    embed_and_store(descriptions)


if __name__ == "__main__":
    main()
