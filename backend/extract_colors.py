"""
Extract dominant colors from clothing items using segmentation masks.

For each image, uses the Fashionpedia polygon segmentation to isolate
the clothing region, then extracts dominant colors via HSV analysis.
Updates ChromaDB entries with color tags and re-embeds descriptions.
"""

import json
import os
import time
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageStat
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ANNOTATIONS_PATH = "data/annotations/instances_attributes_train2020.json"
IMAGES_DIR = "data/images/train"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


def rgb_to_hsv_name(r, g, b):
    """Map an RGB pixel to a human-readable color name using HSV space."""
    r, g, b = int(r) / 255.0, int(g) / 255.0, int(b) / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    diff = mx - mn
    v = mx
    s = 0 if mx == 0 else diff / mx

    # Low saturation = grayscale
    if s < 0.10:
        if v < 0.15: return "black"
        if v < 0.40: return "dark gray"
        if v < 0.60: return "gray"
        if v < 0.80: return "silver"
        if v < 0.92: return "light gray"
        return "white"

    # Compute hue
    if diff == 0:
        h = 0
    elif mx == r:
        h = 60 * (((g - b) / diff) % 6)
    elif mx == g:
        h = 60 * ((b - r) / diff + 2)
    else:
        h = 60 * ((r - g) / diff + 4)

    # Pastels: low saturation + bright
    if s < 0.25 and v > 0.7:
        if 340 <= h or h < 20: return "light pink"
        if 20 <= h < 45: return "beige"
        if 45 <= h < 70: return "cream"
        if 70 <= h < 160: return "light green"
        if 160 <= h < 260: return "light blue"
        if 260 <= h < 340: return "lavender"

    # Saturated colors by hue
    if 340 <= h or h < 10:
        return "pink" if v > 0.6 and s < 0.5 else ("dark red" if v < 0.5 else "red")
    if 10 <= h < 25: return "coral" if v > 0.7 else "brown"
    if 25 <= h < 45: return "orange"
    if 45 <= h < 65: return "yellow" if v > 0.5 else "olive"
    if 65 <= h < 80: return "olive" if v < 0.6 else "yellow"
    if 80 <= h < 160: return "green"
    if 160 <= h < 200: return "teal"
    if 200 <= h < 240: return "blue" if s > 0.4 else "light blue"
    if 240 <= h < 280: return "navy" if v < 0.4 else "purple"
    if 280 <= h < 320: return "purple"
    if 320 <= h < 340: return "pink" if v > 0.5 else "burgundy"
    return "unknown"


def get_mask_pixels(image, segmentation):
    """Create mask from polygon segmentation, return masked pixels."""
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for polygon in segmentation:
        if isinstance(polygon, list) and len(polygon) >= 6:
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=255)

    mask_arr = np.array(mask)
    img_arr = np.array(image)
    return img_arr[mask_arr > 0]


def extract_colors_hsv(pixels, n_colors=3):
    """Extract dominant colors from pixels using HSV voting."""
    if len(pixels) < 10:
        return []

    # Sample for speed
    step = max(1, len(pixels) // 1000)
    sampled = pixels[::step]

    votes = Counter()
    for px in sampled:
        votes[rgb_to_hsv_name(*px)] += 1

    return [c for c, _ in votes.most_common(n_colors)]


def normalize_image(img):
    """Auto-adjust brightness so average luminance is ~128."""
    stat = ImageStat.Stat(img.convert("L"))
    avg = stat.mean[0]
    if avg < 10:
        return img
    factor = max(0.5, min(128.0 / avg, 2.5))
    return ImageEnhance.Brightness(img).enhance(factor)


def main():
    print("Loading annotations...")
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    attributes = {a["id"]: a["name"] for a in data["attributes"]}
    images = {img["id"]: img for img in data["images"]}

    image_annotations = defaultdict(list)
    for ann in data["annotations"]:
        image_annotations[ann["image_id"]].append(ann)

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection("fashion")
    print(f"ChromaDB collection has {collection.count()} items")

    total = len(images)
    updates = []
    errors = 0

    for idx, (image_id, image_info) in enumerate(images.items()):
        file_name = image_info["file_name"]
        image_path = os.path.join(IMAGES_DIR, file_name)

        if not os.path.exists(image_path):
            continue

        anns = image_annotations.get(image_id, [])
        if not anns:
            continue

        try:
            img_raw = Image.open(image_path).convert("RGB")
            img = normalize_image(img_raw)

            all_colors = []
            for ann in anns:
                seg = ann.get("segmentation", [])
                if isinstance(seg, list) and seg:
                    pixels = get_mask_pixels(img, seg)
                    colors = extract_colors_hsv(pixels, n_colors=2)
                    all_colors.extend(colors)

            img_raw.close()
            img.close()

            if not all_colors:
                continue

            color_counter = Counter(all_colors)
            top_colors = [c for c, _ in color_counter.most_common(3)]

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

            parts = []
            parts.append("Colors: " + ", ".join(top_colors))
            if cats:
                parts.append("Items: " + ", ".join(sorted(cats)))
            if attrs:
                parts.append("Attributes: " + ", ".join(sorted(attrs)))
            description = ". ".join(parts)

            updates.append({
                "id": str(image_id),
                "description": description,
                "colors": top_colors,
                "categories": list(cats),
                "attributes": list(attrs),
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error processing {file_name}: {e}")

        if (idx + 1) % 1000 == 0:
            print(f"  Scanned [{idx + 1}/{total}] ({(idx + 1) * 100 // total}%) — {len(updates)} updates queued")

    print(f"\nColor extraction done: {len(updates)} images with colors, {errors} errors")
    print("Now re-embedding and updating ChromaDB...")

    client = OpenAI()
    for i in range(0, len(updates), BATCH_SIZE):
        batch = updates[i : i + BATCH_SIZE]
        texts = [u["description"] for u in batch]

        response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        embeddings = [item.embedding for item in response.data]

        collection.update(
            ids=[u["id"] for u in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "file_name": images[int(u["id"])]["file_name"],
                    "categories": json.dumps(u["categories"]),
                    "attributes": json.dumps(u["attributes"]),
                    "colors": json.dumps(u["colors"]),
                }
                for u in batch
            ],
        )

        processed = min(i + BATCH_SIZE, len(updates))
        print(f"  [{processed}/{len(updates)}] ({processed * 100 // len(updates)}%)")
        if i + BATCH_SIZE < len(updates):
            time.sleep(0.5)

    print(f"\nDone! Updated {len(updates)} entries with color tags.")


if __name__ == "__main__":
    main()
