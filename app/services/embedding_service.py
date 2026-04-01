import logging

from sentence_transformers import SentenceTransformer, CrossEncoder

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_embedding_model: SentenceTransformer | None = None
_reranker_model: CrossEncoder | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        settings = get_settings()
        logger.info("Loading reranker model: %s", settings.reranker_model)
        _reranker_model = CrossEncoder(settings.reranker_model)
    return _reranker_model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def generate_embedding(text: str) -> list[float]:
    return generate_embeddings([text])[0]


def rerank(query: str, documents: list[dict], top_n: int) -> list[dict]:
    if not documents:
        return []

    model = get_reranker_model()
    pairs = [(query, doc["text"]) for doc in documents]
    scores = model.predict(pairs)

    for i, doc in enumerate(documents):
        doc["rerank_score"] = float(scores[i])

    ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
    return ranked[:top_n]
