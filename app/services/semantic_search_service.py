import logging

from app.config.settings import get_settings
from app.models.document import Chunk, Document
from app.services.chunking_service import chunk_document
from app.services.embedding_service import generate_embedding, generate_embeddings, rerank
from app.services.opensearch_connector import (
    delete_by_doc_id,
    ensure_index_exists,
    get_opensearch_client,
)

logger = logging.getLogger(__name__)


def index_documents(documents: list[Document]) -> int:
    ensure_index_exists()
    settings = get_settings()
    client = get_opensearch_client()

    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    texts = [c.text for c in all_chunks]
    embeddings = generate_embeddings(texts)

    for chunk, emb in zip(all_chunks, embeddings):
        chunk.embedding = emb

    for chunk in all_chunks:
        body = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "text": chunk.text,
            "embedding": chunk.embedding,
        }
        client.index(
            index=settings.opensearch_index,
            id=chunk.chunk_id,
            body=body,
            refresh=True,
        )

    logger.info("Indexed %d chunks", len(all_chunks))
    return len(all_chunks)


def search(query: str, top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    if top_k is None:
        top_k = settings.top_k

    query_embedding = generate_embedding(query)

    body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k,
                }
            }
        },
        "_source": ["chunk_id", "doc_id", "title", "text"],
    }

    client = get_opensearch_client()
    resp = client.search(index=settings.opensearch_index, body=body)

    results = []
    for hit in resp["hits"]["hits"]:
        results.append(
            {
                "chunk_id": hit["_source"]["chunk_id"],
                "doc_id": hit["_source"]["doc_id"],
                "title": hit["_source"]["title"],
                "text": hit["_source"]["text"],
                "score": hit["_score"],
            }
        )

    return results


def search_and_rerank(query: str) -> list[dict]:
    settings = get_settings()
    candidates = search(query, top_k=settings.top_k)
    reranked = rerank(query, candidates, top_n=settings.rerank_top_n)
    return reranked


def update_document(document: Document) -> int:
    deleted = delete_by_doc_id(document.id)
    logger.info("Deleted %d old chunks for doc_id=%s", deleted, document.id)
    indexed = index_documents([document])
    return indexed


def delete_document(doc_id: str) -> int:
    deleted = delete_by_doc_id(doc_id)
    logger.info("Deleted %d chunks for doc_id=%s", deleted, doc_id)
    return deleted
