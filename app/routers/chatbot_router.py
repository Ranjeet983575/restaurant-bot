from fastapi import APIRouter, HTTPException

from app.models.document import (
    ChatRequest,
    ChatResponse,
    CreateIndexRequest,
    DeleteRequest,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    UpdateIndexSettingsRequest,
    UpdateRequest,
)
from app.services import chatbot_service, semantic_search_service
from app.services.opensearch_connector import (
    create_index,
    delete_index,
    list_all_indices,
    update_index_settings,
)

router = APIRouter(prefix="/api", tags=["chatbot"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = chatbot_service.chat(request.query)
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest):
    try:
        count = semantic_search_service.index_documents(request.documents)
        return IndexResponse(message="Documents indexed successfully", chunks_indexed=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search(request: SearchRequest):
    try:
        results = semantic_search_service.search(request.query, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/rerank")
def search_rerank(request: SearchRequest):
    try:
        results = semantic_search_service.search_and_rerank(request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/documents/update", response_model=IndexResponse)
def update_document(request: UpdateRequest):
    try:
        count = semantic_search_service.update_document(request.document)
        return IndexResponse(message="Document updated successfully", chunks_indexed=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        deleted = semantic_search_service.delete_document(doc_id)
        return {"message": f"Deleted {deleted} chunks for document {doc_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Index Management Endpoints ───────────────────────────────────────────


@router.post("/indices")
def create_new_index(request: CreateIndexRequest):
    try:
        result = create_index(request.index_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indices")
def get_all_indices():
    try:
        indices = list_all_indices()
        return {"indices": indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/indices/settings")
def update_index(request: UpdateIndexSettingsRequest):
    try:
        result = update_index_settings(request.index_name, request.settings)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/indices/{index_name}")
def remove_index(index_name: str):
    try:
        result = delete_index(index_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Get All Documents in Default Index ─────────────────────────────
from app.services.opensearch_connector import get_all_documents_from_env_index

@router.get("/documents/all")
def get_all_documents():
    try:
        docs = get_all_documents_from_env_index()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
