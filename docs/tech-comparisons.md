# Technology Comparisons

These comparisons were evaluated during the design phase of Fashion-Advisor-RAG.

---

## LLM Provider Comparison

| | Claude (Anthropic) | GPT-4o-mini (OpenAI) | Gemini Flash (Google) |
|---|---|---|---|
| Input cost | $3/1M tokens | $0.15/1M tokens | $0.075/1M tokens |
| Output cost | $15/1M tokens | $0.60/1M tokens | $0.30/1M tokens |
| Best for | High quality understanding | Cost-effective balance | Cheapest option |

**Decision**: GPT-4o-mini — our fashion advice is short (2-3 sentences), so quality difference is minimal but cost difference is significant. For embedding, we use OpenAI's `text-embedding-3-small` at $0.02/1M tokens regardless of chat model choice.

---

## Vector Store Comparison

### What is a vector store?

A vector store (vector database) stores high-dimensional numerical representations (embeddings) of data and enables fast similarity search. In our case, each fashion image's text description is converted to a 1,536-dimension vector. When a user searches "red floral dress", their query is also converted to a vector, and the database finds the closest matching vectors using cosine similarity.

### Options available

| | ChromaDB | Pinecone | Weaviate | FAISS | Milvus | Qdrant |
|---|---|---|---|---|---|---|
| Type | Embedded / Client-server | Fully managed cloud | Self-hosted / Cloud | Library (in-memory) | Self-hosted / Cloud | Self-hosted / Cloud |
| Price | Free | Free tier, then ~$70/mo | Free self-hosted, cloud paid | Free | Free self-hosted | Free self-hosted |
| Setup complexity | `pip install` | Account + API key | Docker + config | `pip install` | Docker + complex setup | Docker + config |
| Python API | Simple, intuitive | Simple | Moderate | Low-level | Moderate | Simple |
| Max vectors (practical) | ~1M | Billions | Billions | ~10M (RAM limited) | Billions | ~100M |
| Persistence | Local file / server | Cloud managed | Disk | Manual save/load | Disk | Disk |
| Filtering (metadata) | Yes | Yes | Yes | No (vectors only) | Yes | Yes |
| Best for | Dev, small-medium projects | Large-scale production | Enterprise, multi-modal | Research, prototypes | Large-scale, high perf | Mid-to-large production |

### Why ChromaDB for this project?

1. **Free** — no cloud costs, no API keys needed
2. **Simplest setup** — `pip install chromadb`, 3 lines of code to start
3. **46K vectors is tiny** — ChromaDB handles this easily; Pinecone/Milvus are designed for millions-to-billions
4. **Metadata filtering** — we store categories, attributes, and colors as metadata alongside vectors
5. **Two deployment modes**:
   - **Embedded** (dev): runs inside your Python process, data stored locally
   - **Client-server** (prod): runs as a separate service, accessible over HTTP
6. **Easy migration** — if we outgrow ChromaDB, switching to Pinecone or Qdrant requires changing ~10 lines of code (the vector store is abstracted behind simple add/query calls)

### Why NOT the others?

| Option | Why we didn't choose it |
|---|---|
| **Pinecone** | Cloud-only, costs money at scale, requires account — overkill for a 46K-item POC |
| **Weaviate** | Powerful but requires Docker + complex config — unnecessary complexity for our simple use case |
| **FAISS** | Facebook's library, very fast, but no metadata filtering (can't store categories/colors alongside vectors), no persistence by default, low-level API |
| **Milvus** | Enterprise-grade, requires Docker + etcd + MinIO — way too heavy for a POC |
| **Qdrant** | Good alternative to ChromaDB, but requires running a Docker container — ChromaDB's embedded mode is simpler for dev |

### When to migrate away from ChromaDB

Consider switching if:
- Data grows beyond ~1M vectors → Pinecone or Milvus
- Need multi-region/high availability → Pinecone (managed)
- Need advanced search (hybrid text+vector) → Weaviate
- Need maximum query speed at scale → Qdrant or FAISS

**Decision**: ChromaDB — free, zero setup, 46K items is well within limits. Can deploy in client-server mode for production. Migration to Pinecone or Qdrant is straightforward if we scale beyond ~1M items.

---

## RAG Framework Comparison

| | LangChain | LlamaIndex | No framework (direct SDK) |
|---|---|---|---|
| Focus | General LLM app framework | RAG-specialized | Manual control |
| Complexity | Heavy, steep learning curve | Medium | Minimal |
| Dependencies | Many | Moderate | Just `openai` + `chromadb` |
| Our use case | Overkill — too many unused features | Good fit but adds abstraction | Perfect fit — simple pipeline |

**What is LangChain?** A general-purpose LLM application framework. It connects to various LLMs, manages prompt templates, calls vector databases, and builds complex agent workflows. Downside: heavy, steep learning curve, many features you won't use.

**What is LlamaIndex?** A framework focused specifically on RAG (Retrieval-Augmented Generation). It handles data indexing and retrieval, with simpler APIs than LangChain. Downside: narrower scope, still adds abstraction.

**Decision**: No framework — our pipeline is straightforward (text → embedding → vector search → return results). Adding LangChain or LlamaIndex would add complexity and dependencies without meaningful benefit. The entire search logic is ~20 lines of code.

---

## Color Detection Approach Comparison

| | Simple RGB matching | HSV-based analysis | OpenAI Vision API |
|---|---|---|---|
| Cost | Free | Free | ~$0.01-0.02/image ($450-900 for 46K) |
| Accuracy | Poor in dim lighting | Good with brightness normalization | Best |
| Speed | Fast | Fast | Slow (API calls) |
| Handles pastels | No (pink → gray) | Yes (pink detected correctly) | Yes |

**Problem**: Fashionpedia dataset has no color annotations — only categories (dress, jacket) and attributes (floral, striped). We needed to add color data ourselves.

**Why RGB failed**: A pink dress in dim lighting had RGB values close to gray. Simple Euclidean distance in RGB space couldn't distinguish pastel colors from gray.

**Why HSV works**: HSV (Hue, Saturation, Value) separates color identity (hue) from brightness (value). Even in dim photos, the hue of pink is preserved. Combined with brightness normalization (auto-adjusting average luminance to 128), this correctly identifies pastel and muted colors.

**Decision**: HSV-based analysis with brightness normalization. OpenAI Vision would be more accurate but costs $450+ for the full dataset — not justified for a POC.

---

## Search Strategy Comparison

| | Pure Vector Search | Vector + Post-filter | LLM Parsing + Vector + Metadata (our approach) |
|---|---|---|---|
| Handles "red dress" | Yes | Yes | Yes |
| Handles "NOT red" | No (vector sees "red" = similar to red) | Partial (can filter after) | Yes (LLM extracts exclusions) |
| Handles "funky style" | Partial (embedding captures some meaning) | Partial | Yes (LLM maps "funky" → asymmetrical, oversized, bold) |
| Category precision | Low (relies on embedding similarity) | Medium (can filter by metadata) | High (LLM extracts exact categories) |
| Color precision | Low | Medium (post-filter by color) | High (filter + boost similar colors) |
| Latency | Fast (1 API call) | Fast (1 API call + local filter) | Slower (2 API calls: parse + embed) |
| Cost per query | ~$0.00002 | ~$0.00002 | ~$0.001 (adds LLM parse call) |

**Problem**: Vector search treats all words equally — "I don't like red" puts the query embedding CLOSE to red items because the word "red" dominates the embedding. Negation is invisible to embeddings.

**Our solution**: A 3-step pipeline:
1. **LLM parses** the natural language into structured JSON (what to search for, what to exclude, what categories/colors/attributes)
2. **Vector search** uses only the positive search terms for embedding similarity
3. **Post-processing** filters out excluded items and re-ranks using metadata bonuses

This adds ~$0.001 per query (one extra GPT-4o-mini call) but dramatically improves search quality for complex queries.

---

## Deployment Platform Comparison

| | Railway | Render | AWS EC2 | Vercel |
|---|---|---|---|---|
| Cost | ~$5/month | Free tier (slow cold starts) | ~$10-30/month | Free (frontend only) |
| Docker support | Yes | Yes | Yes | No (Next.js only) |
| Disk storage | 5GB+ | 0.5GB free | Unlimited | N/A |
| Complexity | Low | Low | High | Very low |
| Our need | 3GB images + 420MB ChromaDB | Insufficient free storage | Overkill for POC | Perfect for frontend |

**Decision**: Railway for backend (handles our 3GB+ data, simple Docker deploy, ~$5/month), Vercel for frontend (free, native Next.js support). AWS was overkill for a POC with limited users. Render's free tier doesn't have enough storage for our images + vector store.

---

## Embedding Model Comparison

| | OpenAI text-embedding-3-small | OpenAI text-embedding-3-large | Local sentence-transformers |
|---|---|---|---|
| Cost | $0.02/1M tokens | $0.13/1M tokens | Free |
| Dimensions | 1,536 | 3,072 | 384-768 |
| Quality | Good | Best | Moderate |
| Setup | API key only | API key only | Download ~100MB model |
| Full dataset cost | ~$1-2 | ~$6-13 | $0 |

**Decision**: OpenAI `text-embedding-3-small` — best cost/quality ratio. The quality difference vs. large model is marginal for fashion text descriptions. Local models save money but produce lower quality embeddings for nuanced fashion attributes.

---

## Production Server Comparison

| | Flask dev server | Gunicorn | uWSGI |
|---|---|---|---|
| Concurrency | 1 request at a time | Multiple workers | Multiple workers |
| Stability | Crashes, memory leaks | Auto-restarts workers | Auto-restarts workers |
| Performance | Slow | 5-10x faster | Similar to Gunicorn |
| Setup | `python app.py` | `gunicorn app:app` | More complex config |
| Flask warning | "Do not use in production" | Recommended | Supported |

**Decision**: Gunicorn — industry standard for Flask, simple one-line config, 2 workers handles ~100-200 requests/min which is plenty for POC. Each worker loads ChromaDB (~420MB RAM), so we keep workers at 2 to stay within Railway's memory limits.
