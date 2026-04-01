import logging


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.chatbot_router import router as chatbot_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


app = FastAPI(
    title="DineSmart AI",
    description="Restaurant chatbot powered by RAG with semantic search",
    version="1.0.0",
)

# Allow all CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "DineSmart AI"}
