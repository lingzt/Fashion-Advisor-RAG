"""
Flask API for Fashion Advisor RAG.

Endpoints:
  POST /api/search  - Search for matching outfits by text description
  POST /api/advice  - Get AI fashion advice based on search results
"""

import json
import os

import chromadb
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=os.getenv("FRONTEND_URL", "*").split(","))

# Config
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
IMAGES_DIR = os.getenv("IMAGES_DIR", "data/images/train")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Initialize clients
openai_client = OpenAI()
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("fashion")

# Color similarity map for boosting and filtering
SIMILAR_COLORS = {
    "red": ["dark red", "crimson", "maroon", "burgundy", "coral"],
    "orange": ["tan", "coral", "brown", "beige"],
    "tan": ["brown", "beige", "orange", "khaki", "cream"],
    "brown": ["tan", "beige", "khaki", "orange"],
    "pink": ["light pink", "hot pink", "coral", "lavender"],
    "blue": ["navy", "light blue", "teal", "royal blue"],
    "navy": ["blue", "dark gray", "black"],
    "green": ["olive", "teal", "light green"],
    "purple": ["lavender", "violet", "magenta", "burgundy"],
    "black": ["dark gray", "charcoal", "navy"],
    "white": ["cream", "ivory", "light gray", "silver"],
    "gray": ["silver", "light gray", "dark gray", "charcoal"],
    "yellow": ["gold", "cream", "beige"],
    "beige": ["tan", "cream", "ivory", "khaki", "brown"],
    "teal": ["blue", "green", "turquoise"],
}

# LLM system prompt for structured query parsing
PARSE_SYSTEM_PROMPT = """You parse natural language fashion queries into a structured JSON search filter.

Available metadata fields in our database:

CATEGORIES (clothing items):
shirt/blouse, top/t-shirt/sweatshirt, sweater, cardigan, jacket, vest, pants, shorts, skirt, coat, dress, jumpsuit, cape, glasses, hat, headband/hair accessory, tie, glove, watch, belt, leg warmer, tights/stockings, sock, shoe, bag/wallet, scarf, umbrella

COLORS (detected from images):
red, dark red, pink, hot pink, light pink, orange, yellow, gold, brown, tan, beige, cream, ivory, green, olive, teal, blue, navy, light blue, purple, lavender, white, light gray, gray, dark gray, charcoal, black, silver, khaki, burgundy, coral, salmon

ATTRIBUTES - patterns:
plain, abstract, camouflage, check, dot, floral, geometric, stripe, houndstooth, paisley, plaid

ATTRIBUTES - styles/silhouettes:
symmetrical, asymmetrical, peplum, flare, fit and flare, mermaid, a-line, straight, baggy, oversized, slim

ATTRIBUTES - length:
micro, mini, above-the-knee, knee, below the knee, midi, maxi, floor

ATTRIBUTES - materials:
leather, suede, denim, lace, silk, velvet, knit, fur, feather

Parse the user's query and respond with ONLY this JSON:
{
  "search_text": "a concise description of what they want, optimized for embedding search",
  "want_categories": ["matching category names from the list above"],
  "want_colors": ["colors they want"],
  "want_attributes": ["patterns, styles, materials they want"],
  "exclude_colors": ["colors they do NOT want"],
  "exclude_attributes": ["patterns/styles they do NOT want"],
  "occasion": "occasion if mentioned (wedding, casual, work, party, etc.)"
}

Only include fields that the user actually mentioned. Leave arrays empty if not specified.
Be smart about mapping — "funky" could mean asymmetrical, oversized, bold patterns. "Elegant" could mean symmetrical, floor length. Include these interpretations in search_text."""


@app.route("/api/search", methods=["POST"])
def search():
    """Search for matching fashion images using LLM-parsed structured query."""
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Step 1: LLM parses natural language into structured query
    parse_response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(parse_response.choices[0].message.content)

    search_text = parsed.get("search_text", query)
    want_colors = [c.lower() for c in parsed.get("want_colors", [])]
    exclude_colors = [c.lower() for c in parsed.get("exclude_colors", [])]
    want_categories = parsed.get("want_categories", [])
    want_attributes = [a.lower() for a in parsed.get("want_attributes", [])]
    exclude_attributes = [a.lower() for a in parsed.get("exclude_attributes", [])]

    # Step 2: Build embedding query — boost wanted colors
    boosted_query = search_text
    if want_colors:
        all_colors = list(want_colors)
        for c in want_colors:
            all_colors.extend(SIMILAR_COLORS.get(c, []))
        boost = " ".join(all_colors)
        boosted_query = f"{search_text}. Color: {boost} {boost}"

    # Step 3: Generate embedding
    response = openai_client.embeddings.create(
        input=[boosted_query], model=EMBEDDING_MODEL
    )
    query_embedding = response.data[0].embedding

    # Step 4: Build ChromaDB where filter for categories
    where_filter = None
    if want_categories:
        # Match documents that contain any of the wanted categories
        if len(want_categories) == 1:
            where_filter = {"$contains": want_categories[0].lower()}
        # ChromaDB document search handles multiple terms via embedding

    # Fetch extra results to allow for post-filtering
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=48,
        where_document=where_filter,
    )

    # Step 5: Build exclude sets (expand with similar colors)
    full_exclude_colors = set(exclude_colors)
    for c in exclude_colors:
        full_exclude_colors.update(SIMILAR_COLORS.get(c, []))

    # Step 6: Post-filter and rank results
    items = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        item_colors = json.loads(metadata.get("colors", "[]"))
        item_colors_lower = [c.lower() for c in item_colors]
        item_categories = json.loads(metadata.get("categories", "[]"))
        item_categories_lower = [c.lower() for c in item_categories]
        item_attrs = json.loads(metadata.get("attributes", "[]"))
        item_attrs_lower = [a.lower() for a in item_attrs]
        doc = results["documents"][0][i].lower()

        # Filter: skip if primary color is excluded
        if full_exclude_colors and item_colors_lower:
            if item_colors_lower[0] in full_exclude_colors:
                continue

        # Filter: skip if excluded attributes are present
        if exclude_attributes:
            if any(ea in doc for ea in exclude_attributes):
                continue

        distance = results["distances"][0][i]
        base_score = 1 - distance

        # Bonus: boost score if item matches wanted categories
        category_bonus = 0
        if want_categories:
            for wc in want_categories:
                if any(wc.lower() in ic for ic in item_categories_lower):
                    category_bonus += 0.05

        # Bonus: boost if wanted colors match
        color_bonus = 0
        if want_colors:
            want_expanded = set(want_colors)
            for c in want_colors:
                want_expanded.update(SIMILAR_COLORS.get(c, []))
            matching = set(item_colors_lower) & want_expanded
            color_bonus = len(matching) * 0.03

        # Bonus: boost if wanted attributes match
        attr_bonus = 0
        if want_attributes:
            for wa in want_attributes:
                if wa in doc:
                    attr_bonus += 0.02

        final_score = round(base_score + category_bonus + color_bonus + attr_bonus, 3)

        items.append({
            "id": results["ids"][0][i],
            "score": final_score,
            "file_name": metadata["file_name"],
            "image_url": f"/api/images/{metadata['file_name']}",
            "categories": item_categories,
            "attributes": item_attrs,
            "colors": item_colors,
            "description": results["documents"][0][i],
        })

    # Sort by final score (with bonuses) and return top 12
    items.sort(key=lambda x: x["score"], reverse=True)
    items = items[:12]

    return jsonify({"query": query, "parsed": parsed, "results": items})


@app.route("/api/advice", methods=["POST"])
def advice():
    """Generate AI fashion advice based on query and search results."""
    data = request.get_json()
    query = data.get("query", "").strip()
    results = data.get("results", [])

    if not query:
        return jsonify({"error": "Query is required"}), 400

    context = "Here are the top matching fashion items found:\n\n"
    for i, item in enumerate(results[:12], 1):
        context += f"{i}. Categories: {', '.join(item.get('categories', []))}\n"
        context += f"   Colors: {', '.join(item.get('colors', []))}\n"
        context += f"   Attributes: {', '.join(item.get('attributes', []))}\n"
        context += f"   Match score: {item.get('score', 0)}\n\n"

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional fashion advisor. Based on the user's "
                    "description and the matching items found in the fashion database, "
                    "provide brief, helpful styling advice. Keep it concise (2-3 sentences). "
                    "Be friendly and encouraging."
                ),
            },
            {
                "role": "user",
                "content": f"I'm looking for: {query}\n\n{context}\nPlease give me styling advice based on these results.",
            },
        ],
        max_tokens=200,
    )

    advice_text = response.choices[0].message.content
    return jsonify({"advice": advice_text})


@app.route("/api/images/<path:filename>")
def serve_image(filename):
    """Serve fashion images."""
    return send_from_directory(IMAGES_DIR, filename)


if __name__ == "__main__":
    print(f"Fashion collection has {collection.count()} items")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(debug=debug, host="0.0.0.0", port=port)
