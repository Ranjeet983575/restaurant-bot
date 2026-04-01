import logging

from opensearchpy import OpenSearch

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_client: OpenSearch | None = None

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def get_opensearch_client() -> OpenSearch:
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    _client = OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )
    return _client


def ensure_index_exists() -> None:
    settings = get_settings()
    client = get_opensearch_client()
    index = settings.opensearch_index

    if client.indices.exists(index=index):
        logger.info("Index '%s' already exists", index)
        return

    body = {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "faiss",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
            }
        },
    }

    client.indices.create(index=index, body=body)
    logger.info("Created index '%s'", index)


def delete_by_doc_id(doc_id: str) -> int:
    settings = get_settings()
    client = get_opensearch_client()
    resp = client.delete_by_query(
        index=settings.opensearch_index,
        body={"query": {"term": {"doc_id": doc_id}}},
        refresh=True,
    )
    return resp.get("deleted", 0)


def create_index(index_name: str) -> dict:
    client = get_opensearch_client()

    if client.indices.exists(index=index_name):
        return {"message": f"Index '{index_name}' already exists", "created": False}

    body = {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": EMBEDDING_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "faiss",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
            }
        },
    }

    client.indices.create(index=index_name, body=body)
    logger.info("Created index '%s'", index_name)
    return {"message": f"Index '{index_name}' created successfully", "created": True}


def update_index_settings(index_name: str, settings_body: dict) -> dict:
    client = get_opensearch_client()

    if not client.indices.exists(index=index_name):
        raise ValueError(f"Index '{index_name}' does not exist")

    # Close index to apply certain settings, then reopen
    client.indices.close(index=index_name)
    try:
        client.indices.put_settings(index=index_name, body=settings_body)
    finally:
        client.indices.open(index=index_name)

    logger.info("Updated settings for index '%s'", index_name)
    return {"message": f"Index '{index_name}' settings updated successfully"}


def list_all_indices() -> list[dict]:
    client = get_opensearch_client()
    indices = client.cat.indices(format="json")
    result = []
    for idx in indices:
        result.append({
            "index": idx.get("index"),
            "health": idx.get("health"),
            "status": idx.get("status"),
            "docs_count": idx.get("docs.count"),
            "store_size": idx.get("store.size"),
        })
    return result


def delete_index(index_name: str) -> dict:
    client = get_opensearch_client()

    if not client.indices.exists(index=index_name):
        return {"message": f"Index '{index_name}' does not exist", "deleted": False}

    client.indices.delete(index=index_name)
    logger.info("Deleted index '%s'", index_name)
    return {"message": f"Index '{index_name}' deleted successfully", "deleted": True}


# Fetch all documents from the default index (from env)
def get_all_documents_from_env_index() -> list[dict]:
    settings = get_settings()
    client = get_opensearch_client()
    index = settings.opensearch_index
    # Use a scroll search for large datasets, but for demo, fetch up to 1000
    resp = client.search(index=index, body={"query": {"match_all": {}}}, size=1000)
    docs = []
    for hit in resp["hits"]["hits"]:
        doc = hit["_source"]
        doc["_id"] = hit["_id"]
        docs.append(doc)
    return docs
