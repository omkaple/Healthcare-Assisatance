"""
FastAPI application - Healthcare AI Assistant API.
Exposes endpoints for document ingestion, question answering, and health checks.
"""

import logging
import time
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.rag import ingest_documents, get_vector_store_stats
from app.agent import handle_query

# ─── Configure Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── FastAPI App ───
app = FastAPI(
    title="Healthcare AI Assistant",
    description=(
        "A RAG-based AI assistant for cancer-related healthcare information. "
        "Answers questions using only the provided knowledge base documents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS Middleware ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ───
class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="The healthcare question to answer",
        examples=["What are the treatment options for breast cancer?"],
    )
    chat_history: list[ChatMessage] = Field(
        default=[],
        description="Previous conversation messages for context",
    )


class SourceInfo(BaseModel):
    """Source citation information."""
    document: str
    chunk: str


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str
    sources: list[SourceInfo]
    confidence: str
    workflow: str
    response_time_ms: float


class IngestResponse(BaseModel):
    """Response model for the /ingest endpoint."""
    message: str
    chunks_ingested: int
    total_chunks_in_store: int


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str
    vector_store: dict


# ─── Endpoints ───

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint. Returns system status and vector store stats."""
    logger.info("Health check requested.")
    try:
        stats = get_vector_store_stats()
        return HealthResponse(status="ok", vector_store=stats)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="degraded", vector_store={"error": str(e)})


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
def ingest():
    """
    Ingest healthcare documents from the /data folder.
    Reads documents, splits into chunks, generates embeddings,
    and stores them in the ChromaDB vector database.
    """
    logger.info("Document ingestion requested.")
    try:
        num_chunks = ingest_documents()
        stats = get_vector_store_stats()
        msg = f"Successfully ingested {num_chunks} chunks from documents."
        logger.info(msg)
        return IngestResponse(
            message=msg,
            chunks_ingested=num_chunks,
            total_chunks_in_store=stats["total_chunks"],
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=AskResponse, tags=["Question Answering"])
def ask(request: AskRequest):
    """
    Ask a healthcare question. The assistant uses RAG to answer
    from the ingested knowledge base, or routes to mock tools
    for appointment-related queries.
    """
    start = time.time()
    logger.info(f"Question received: '{request.question}'")

    try:
        result = handle_query(
            request.question,
            chat_history=[m.model_dump() for m in request.chat_history],
        )
        elapsed = (time.time() - start) * 1000

        logger.info(f"Response generated in {elapsed:.1f}ms (confidence: {result['confidence']})")

        return AskResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            confidence=result["confidence"],
            workflow=result["workflow"],
            response_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


# ─── Serve Chatbot UI ───
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def root():
    """Serve the chatbot UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Startup Event ───
@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 60)
    logger.info("Healthcare AI Assistant starting up...")
    logger.info("Chat UI at: http://localhost:8000")
    logger.info("API Docs at: http://localhost:8000/docs")
    logger.info("=" * 60)
