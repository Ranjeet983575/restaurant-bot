from app.config.settings import get_settings
from app.models.document import Chunk, Document


def chunk_document(document: Document) -> list[Chunk]:
    settings = get_settings()
    text = document.content
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Avoid dangling tiny fragments
        if len(chunk_text.strip()) == 0:
            break

        chunks.append(
            Chunk(
                chunk_id=f"{document.id}_chunk_{idx}",
                doc_id=document.id,
                title=document.title,
                text=chunk_text,
            )
        )
        idx += 1
        start += chunk_size - overlap

    return chunks
