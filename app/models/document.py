from pydantic import BaseModel


class Document(BaseModel):
    id: str
    title: str
    content: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    embedding: list[float] | None = None


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


class IndexRequest(BaseModel):
    documents: list[Document]


class IndexResponse(BaseModel):
    message: str
    chunks_indexed: int


class UpdateRequest(BaseModel):
    document: Document


class DeleteRequest(BaseModel):
    doc_id: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class CreateIndexRequest(BaseModel):
    index_name: str


class UpdateIndexSettingsRequest(BaseModel):
    index_name: str
    settings: dict
