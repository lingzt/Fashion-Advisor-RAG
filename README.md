# Fashion Advisor RAG

A RAG-based fashion advisor that takes a text description and returns the top 5 matching outfit images from the Fashionpedia dataset, along with AI-generated styling advice.

## Architecture

```
User Browser
    |
Next.js Frontend (Vercel)
    |
Flask API (Railway)
    |
ChromaDB (embedded, same server)
```

## Key Technical Decisions

### How search works (LLM + Vector + Metadata)
The search pipeline combines three techniques:

1. **LLM Query Parsing** — GPT-4o-mini converts natural language into structured JSON:
   ```
   "I need a funky dress for a wedding, no red"
   →  {
        "want_categories": ["dress"],
        "want_attributes": ["asymmetrical", "oversized"],
        "exclude_colors": ["red"],
        "occasion": "wedding",
        "search_text": "funky dress for wedding with bold patterns"
      }
   ```

2. **Vector Similarity Search** — The `search_text` is embedded and matched against ChromaDB (cosine similarity). Color terms are boosted with similar colors (e.g., "blue" also boosts "navy", "teal").

3. **Metadata Filtering & Re-ranking** — Results are post-filtered (excluded colors/attributes removed) and re-ranked with bonuses for matching categories (+5%), colors (+3%), and attributes (+2%).

This handles complex queries like "casual blue denim jacket, not too dark, no leather" that pure vector search cannot.

### Why ChromaDB over Pinecone?
- **Free and local** — no cloud costs, no API key needed for the vector store
- Fashionpedia has ~46K images — well within ChromaDB's capacity
- Runs embedded in the same process as Flask, zero network latency for queries
- Easy migration to Pinecone later if needed (just swap the client)

### Why OpenAI `text-embedding-3-small` for embeddings?
- Cheapest option at $0.02/1M tokens
- 1536 dimensions — good balance of quality vs storage
- Alternatives considered: local sentence-transformers (free but lower quality), Claude Voyager embeddings
- Full dataset embedding cost: ~$1-2 one-time

### Why GPT-4o-mini for fashion advice?
- Cheapest chat model at $0.15/1M input tokens
- Only used for short fashion advice (2-3 sentences), so cost per query is negligible
- Alternatives considered: Claude (higher quality but 20x cost for this use case), no LLM (just show images)

### Why Flask over FastAPI?
- Simpler for a single-page app with 3 endpoints
- User preference — familiar stack
- Gunicorn as production WSGI server handles concurrency

### Why Next.js for a single page?
- User preference for Node.js frontend
- Easy deployment to Vercel (free)
- Could have been a plain HTML file, but Next.js gives us environment variables and easy builds

### Why Railway for backend hosting?
- ~$5/month — cheapest option that handles 3GB+ of image data + 420MB ChromaDB
- Auto-deploys from GitHub
- Supports Docker and persistent storage
- Alternative considered: Render (similar but slower cold starts)

### Why HSV-based color detection?
- Fashionpedia dataset has no color annotations — only categories and attributes (e.g., "floral", "striped")
- We extract colors from clothing regions using segmentation masks + HSV color space analysis
- RGB-based detection failed on dimly lit photos — HSV handles brightness variations better
- Brightness normalization applied before color extraction for consistency

### Why color similarity boosting in search?
- Users searching "orange jacket" should also see tan/brown jackets ranked highly
- A color similarity map expands queries: "orange" also boosts "tan", "coral", "brown"
- This happens at query time (no re-indexing needed)

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend API | Python + Flask | Simple, familiar |
| Vector Store | ChromaDB (local) | Free, embedded, sufficient for 46K items |
| Embeddings | OpenAI text-embedding-3-small | Cheapest, good quality |
| LLM | GPT-4o-mini | Cheapest chat model |
| Frontend | Next.js (Pages Router) | Easy Vercel deploy |
| Production Server | Gunicorn | Multi-worker, production-grade |
| Color Detection | PIL + HSV analysis | Free, local, uses segmentation masks |
| Dataset | Fashionpedia (CVDF) | 46K images, rich annotations |

For detailed side-by-side comparisons of each technology choice, see [docs/tech-comparisons.md](docs/tech-comparisons.md).

## Project Structure

```
Fashion-Advisor-RAG/
├── backend/
│   ├── app.py              # Flask API (search + advice + serve images)
│   ├── ingest.py           # One-time: process Fashionpedia → ChromaDB
│   ├── extract_colors.py   # One-time: add color tags to ChromaDB entries
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── Procfile            # Railway deployment
│   ├── .env                # OPENAI_API_KEY (not committed)
│   ├── chroma_db/          # Vector store (420MB, not committed)
│   └── data/
│       ├── images/train/   # 46K fashion images (not committed)
│       └── annotations/    # Fashionpedia JSON (not committed)
├── frontend/
│   ├── pages/index.js      # Single-page chat UI
│   ├── .env.local          # NEXT_PUBLIC_API_URL
│   └── package.json
├── .gitignore
└── README.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/search` | Search for top 5 matching outfits. Body: `{"query": "red floral dress"}` |
| POST | `/api/advice` | Get AI styling advice. Body: `{"query": "...", "results": [...]}` |
| GET | `/api/images/<filename>` | Serve a fashion image |

## Local Development

### Prerequisites
- Python 3.12
- Node.js
- OpenAI API key

### Backend
```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Add your OPENAI_API_KEY
python app.py           # Runs on http://localhost:5000
```

### Data Ingestion (one-time)
```bash
# Download Fashionpedia dataset first (see Data Setup below)
python ingest.py            # ~10 min, embeds all images into ChromaDB
python extract_colors.py    # ~30 min, adds color tags
```

### Frontend
```bash
cd frontend
npm install
npm run dev    # Runs on http://localhost:3000
```

### Data Setup
Download from [Fashionpedia CVDF](https://github.com/cvdfoundation/fashionpedia):
1. Training images → `backend/data/images/train/`
2. `instances_attributes_train2020.json` → `backend/data/annotations/`
3. `attributes_train2020.json` → `backend/data/annotations/`

## Deployment

### Backend → Railway
1. Push to GitHub
2. Connect repo to [Railway](https://railway.app)
3. Set environment variables: `OPENAI_API_KEY`, `FRONTEND_URL`
4. Railway auto-detects Dockerfile

### Frontend → Vercel
1. Connect repo to [Vercel](https://vercel.com)
2. Set `NEXT_PUBLIC_API_URL` to your Railway backend URL
3. Auto-deploys on push

## Cost Breakdown

| Item | Cost |
|------|------|
| Railway (backend hosting) | ~$5/month |
| Vercel (frontend hosting) | Free |
| OpenAI embeddings (one-time ingestion) | ~$1-2 |
| OpenAI per search query | ~$0.001 |
| OpenAI per advice response | ~$0.002 |
| ChromaDB | Free (embedded) |
| **Total for POC** | **~$5/month + pennies per query** |
