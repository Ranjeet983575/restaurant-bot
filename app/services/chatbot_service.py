import logging

from groq import Groq

from app.config.settings import get_settings
from app.services.semantic_search_service import search_and_rerank

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are DineSmart AI, a helpful restaurant assistant.

Instructions:
- Answer only from the provided context
- Be friendly and helpful
- Suggest dishes when relevant
- Keep answers short and clear
- If not found, say "I don't know"

Context:
{context}

User Question:
{query}"""

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        settings = get_settings()
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for c in chunks:
        parts.append(f"[{c['title']}]: {c['text']}")
    return "\n\n".join(parts)


def chat(query: str) -> dict:
    chunks = search_and_rerank(query)

    context = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    settings = get_settings()
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )

    answer = response.choices[0].message.content

    sources = [
        {"chunk_id": c["chunk_id"], "title": c["title"], "score": round(c.get("rerank_score", 0), 4)}
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}
