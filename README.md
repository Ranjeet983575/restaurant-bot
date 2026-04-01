# 🍽️ DineSmart AI — Restaurant Chatbot (RAG + Semantic Search)

A production-ready restaurant chatbot powered by **Retrieval-Augmented Generation (RAG)** with semantic search, reranking, and LLM-based answer generation.

---

## 📐 Architecture

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│   FastAPI Server  │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│              RAG Pipeline                         │
│                                                   │
│  1. Query → Embedding  (SentenceTransformer)      │
│  2. KNN Search         (OpenSearch knn_vector)    │
│  3. Rerank Results     (CrossEncoder)             │
│  4. Build Context      (Top-N chunks)             │
│  5. Generate Answer    (Groq LLM)                 │
└──────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   Response   │
│  answer +    │
│  sources     │
└──────────────┘
```

### Data Indexing Flow

```
┌────────────┐    ┌────────────┐    ┌─────────────────┐    ┌────────────┐
│  Document   │───▶│  Chunking  │───▶│  Embeddings     │───▶│ OpenSearch  │
│  (JSON)     │    │ (200/50)   │    │ (MiniLM-L6-v2)  │    │ (knn_vector)│
└────────────┘    └────────────┘    └─────────────────┘    └────────────┘
```

---

## 🧰 Tech Stack

| Component          | Technology                                |
|--------------------|-------------------------------------------|
| Backend Framework  | FastAPI                                   |
| Vector Database    | OpenSearch 2.18 (knn_vector + HNSW)       |
| Embedding Model    | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Reranker           | `cross-encoder/ms-marco-MiniLM-L-6-v2`   |
| LLM                | Groq (`llama-3.1-8b-instant`)             |
| Build Tool         | uv                                        |
| Language           | Python 3.12+                              |

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed
- Docker & Docker Compose
- A [Groq API key](https://console.groq.com/)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd restaurant-bot
```

### 2. Start OpenSearch (Docker)

```bash
docker compose up -d
```

Wait ~30 seconds for OpenSearch to initialize. Verify it's running:

```bash
curl -ku admin:Admin@1234 https://localhost:9200
```

You should see a JSON response with the OpenSearch cluster info.

### 3. Create Virtual Environment & Install Dependencies

```bash
uv venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
uv sync
```

### 4. Configure Environment Variables

Copy the example env file and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env` and set your `GROQ_API_KEY`:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 5. Seed the Database

Index the sample restaurant data into OpenSearch:

```bash
uv run python seed_data.py
```

### 6. Start the FastAPI Server

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`.

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 API Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{"status": "healthy", "service": "DineSmart AI"}
```

---

### Chat (Main Endpoint)

```
POST /api/chat
```

**Request:**
```json
{
  "query": "What Italian dishes do you have?"
}
```

**Response:**
```json
{
  "answer": "We have a wonderful Italian menu! Our offerings include pasta, pizza, lasagna, and risotto. Some popular dishes are Margherita Pizza, Alfredo Pasta, and Spaghetti Bolognese. We also have gluten-free pasta and vegan pizza options!",
  "sources": [
    {"chunk_id": "2_chunk_0", "title": "Italian Menu", "score": 0.9812},
    {"chunk_id": "1_chunk_0", "title": "Restaurant Overview", "score": 0.6543}
  ]
}
```

---

### Index Documents

```
POST /api/index
```

**Request:**
```json
{
  "documents": [
    {
      "id": "9",
      "title": "New Specials",
      "content": "Try our new weekend brunch menu featuring eggs benedict and avocado toast."
    }
  ]
}
```

**Response:**
```json
{"message": "Documents indexed successfully", "chunks_indexed": 1}
```

---

### Semantic Search

```
POST /api/search
```

**Request:**
```json
{
  "query": "Do you have vegan food?",
  "top_k": 3
}
```

---

### Semantic Search with Reranking

```
POST /api/search/rerank
```

**Request:**
```json
{
  "query": "What desserts are available?"
}
```

---

### Update a Document

```
PUT /api/documents/update
```

**Request:**
```json
{
  "document": {
    "id": "4",
    "title": "Desserts and Beverages",
    "content": "We offer coffee, juices, soft drinks, chocolate cake, ice cream, cheesecake, brownies, and our new tiramisu."
  }
}
```

---

### Delete a Document

```
DELETE /api/documents/{doc_id}
```

**Example:**
```
DELETE /api/documents/4
```

**Response:**
```json
{"message": "Deleted 1 chunks for document 4"}
```

---

### Create an Index

```
POST /api/indices
```

**Request:**
```json
{
  "index_name": "my_new_index"
}
```

**Response:**
```json
{"message": "Index 'my_new_index' created successfully", "created": true}
```

---

### Get All Indices

```
GET /api/indices
```

**Response:**
```json
{
  "indices": [
    {
      "index": "restaurant_chunks",
      "health": "green",
      "status": "open",
      "docs_count": "10",
      "store_size": "156.2kb"
    }
  ]
}
```

---

### Update Index Settings

```
PUT /api/indices/settings
```

**Request:**
```json
{
  "index_name": "restaurant_chunks",
  "settings": {
    "index": {
      "number_of_replicas": 2
    }
  }
}
```

**Response:**
```json
{"message": "Index 'restaurant_chunks' settings updated successfully"}
```

---

### Delete an Index

```
DELETE /api/indices/{index_name}
```

**Example:**
```
DELETE /api/indices/my_new_index
```

**Response:**
```json
{"message": "Index 'my_new_index' deleted successfully", "deleted": true}
```

---

## 🔍 How Semantic Search Works

1. **Chunking**: Each document is split into overlapping chunks (size=200 chars, overlap=50 chars) to preserve context across boundaries.

2. **Embedding**: Each chunk is converted to a 384-dimensional vector using the `all-MiniLM-L6-v2` SentenceTransformer model.

3. **Indexing**: Vectors are stored in OpenSearch using the `knn_vector` field type with HNSW (Hierarchical Navigable Small World) indexing for fast approximate nearest-neighbor search.

4. **Query**: The user's query is embedded using the same model, and OpenSearch performs a KNN search to find the most semantically similar chunks.

5. **Reranking**: The top-K results are passed through a CrossEncoder (`ms-marco-MiniLM-L-6-v2`) which scores each (query, chunk) pair more accurately than embedding similarity alone.

---

## 🤖 How the Chatbot Works (RAG Pipeline)

```
User Query
    │
    ▼
[Embed Query] ──▶ [KNN Search in OpenSearch] ──▶ [Top-K Chunks]
                                                        │
                                                        ▼
                                                [CrossEncoder Rerank]
                                                        │
                                                        ▼
                                                [Top-N Best Chunks]
                                                        │
                                                        ▼
                                              [Build Prompt Context]
                                                        │
                                                        ▼
                                                [Groq LLM generates answer]
                                                        │
                                                        ▼
                                              [Return answer + sources]
```

1. The user sends a natural language question.
2. The query is converted to an embedding vector.
3. OpenSearch finds the top-K most similar chunks via KNN search.
4. A CrossEncoder reranks these chunks for higher accuracy.
5. The top-N chunks are assembled into a context block.
6. A prompt is constructed with the context and user question.
7. Groq's LLM generates a grounded answer from the context only.
8. The answer and source chunks are returned to the user.

---

## 💬 Example Queries & Expected Responses

| Query | Expected Response |
|-------|-------------------|
| "What Italian dishes do you have?" | Lists pasta, pizza, lasagna, risotto, and popular dishes |
| "Do you deliver?" | Explains 5 km delivery radius and online ordering |
| "What are the opening hours?" | "Open daily from 10 AM to 10 PM. Peak hours are 7 PM to 9 PM." |
| "Do you have vegan options?" | Lists vegan meals, salads, smoothie bowls, plant-based dishes |
| "Can I reserve a table?" | Explains online/phone reservations, suggests weekends |
| "What desserts do you serve?" | Lists chocolate cake, ice cream, cheesecake, brownies, etc. |
| "Do you have sushi?" | Mentions fresh sushi and sashimi prepared daily |
| "What is the weather today?" | "I don't know" (not in restaurant context) |

---

## 📁 Project Structure

```
restaurant-bot/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI app entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                  # Pydantic settings (env-based)
│   ├── models/
│   │   ├── __init__.py
│   │   └── document.py                  # Request/response models
│   ├── routers/
│   │   ├── __init__.py
│   │   └── chatbot_router.py            # API route handlers
│   └── services/
│       ├── __init__.py
│       ├── chatbot_service.py           # LLM prompt + Groq integration
│       ├── chunking_service.py          # Document chunking logic
│       ├── embedding_service.py         # Embedding + reranking
│       ├── opensearch_connector.py      # OpenSearch client + index mgmt
│       └── semantic_search_service.py   # Search, rerank, index, CRUD
├── seed_data.py                         # Seed script for sample data
├── docker-compose.yml                   # OpenSearch + Dashboards
├── pyproject.toml                       # UV project config
├── .env.example                         # Environment variable template
└── README.md                            # This file
```

---

## 🛠️ Development

### Run tests (curl examples)

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What Italian dishes do you have?"}'

# Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vegan food", "top_k": 3}'

# Index new document
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"id": "9", "title": "Specials", "content": "Weekend brunch menu with eggs benedict."}]}'

# Delete document
curl -X DELETE http://localhost:8000/api/documents/9

# Create a new index
curl -X POST http://localhost:8000/api/indices \
  -H "Content-Type: application/json" \
  -d '{"index_name": "my_new_index"}'

# List all indices
curl http://localhost:8000/api/indices

# Update index settings
curl -X PUT http://localhost:8000/api/indices/settings \
  -H "Content-Type: application/json" \
  -d '{"index_name": "restaurant_chunks", "settings": {"index": {"number_of_replicas": 2}}}'

# Delete an index
curl -X DELETE http://localhost:8000/api/indices/my_new_index
```

---

## 📄 License

MIT
