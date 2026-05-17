"""
RAG (Retrieval-Augmented Generation) pipeline module.
Handles document ingestion, chunking, embedding, storage, and retrieval.
"""

import os
import logging
import chromadb
from app.config import settings
from app.embeddings import generate_embeddings

logger = logging.getLogger(__name__)

_chroma_client = None
_collection = None


def _get_chroma_collection():
    """Get or create the ChromaDB collection. Supports cloud and local modes."""
    global _chroma_client, _collection
    if _collection is None:
        if settings.CHROMA_MODE == "cloud" and settings.CHROMA_API_KEY:
            logger.info("Initializing ChromaDB Cloud client...")
            _chroma_client = chromadb.CloudClient(
                tenant=settings.CHROMA_TENANT,
                database=settings.CHROMA_DATABASE,
                api_key=settings.CHROMA_API_KEY,
            )
            logger.info(f"Connected to ChromaDB Cloud (tenant: {settings.CHROMA_TENANT})")
        else:
            logger.info(f"Initializing local ChromaDB at: {settings.VECTOR_STORE_DIR}")
            os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)

        _collection = _chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{settings.COLLECTION_NAME}' ready. Count: {_collection.count()}")
    return _collection


def _load_documents(data_dir: str) -> list[dict]:
    """Load all .txt files from the data directory."""
    documents = []
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory does not exist: {data_dir}")
        return documents

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            if filename.lower().endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append({"content": content, "source": filename})
                logger.info(f"Loaded: {filename} ({len(content)} chars)")
            elif filename.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(filepath)
                    text = ""
                    for page in reader.pages:
                        pt = page.extract_text()
                        if pt:
                            text += pt + "\n"
                    if text.strip():
                        documents.append({"content": text.strip(), "source": filename})
                        logger.info(f"Loaded PDF: {filename}")
                except Exception as e:
                    logger.error(f"Error reading PDF {filename}: {e}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def _split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            search_start = max(start + int(chunk_size * 0.8), start)
            best_break = -1
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                pos = text.rfind(sep, search_start, end)
                if pos > best_break:
                    best_break = pos + len(sep)
            if best_break > start:
                end = best_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start >= len(text):
            break
    return chunks


def ingest_documents() -> int:
    """Ingest documents from data directory into vector store. Returns chunk count."""
    collection = _get_chroma_collection()
    documents = _load_documents(settings.DATA_DIR)

    if not documents:
        logger.warning("No documents found to ingest.")
        return 0

    all_chunks, all_metadatas, all_ids = [], [], []

    for doc in documents:
        chunks = _split_into_chunks(doc["content"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['source']}__chunk_{i}"
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc["source"], "chunk_index": i, "total_chunks": len(chunks)})
            all_ids.append(chunk_id)

    if not all_chunks:
        return 0

    logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = generate_embeddings(all_chunks)

    batch_size = 1000
    overlapping_chunks_amount = 350
    for i in range(0, len(all_chunks), batch_size - overlapping_chunks_amount):
        end = min(i + batch_size, len(all_chunks))
        collection.upsert(
            ids=all_ids[i:end], embeddings=embeddings[i:end],
            documents=all_chunks[i:end], metadatas=all_metadatas[i:end],
        )
    total = collection.count()
    logger.info(f"Ingestion complete. Total chunks: {total}")
    return len(all_chunks)


def retrieve_relevant_chunks(query: str, top_k: int = None) -> list[dict]:
    """Retrieve most relevant document chunks for a query."""
    if top_k is None:
        top_k = settings.TOP_K_RESULTS
    collection = _get_chroma_collection()
    if collection.count() == 0:
        logger.warning("Vector store is empty.")
        return []

    query_embedding = generate_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding], n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if results and results["documents"]:
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            retrieved.append({
                "document": meta.get("source", "unknown"),
                "chunk": doc, "distance": dist,
                "chunk_index": meta.get("chunk_index", 0),
            })
    logger.info(f"Retrieved {len(retrieved)} chunks for: '{query[:60]}...'")
    return retrieved


def get_vector_store_stats() -> dict:
    """Get statistics about the vector store."""
    collection = _get_chroma_collection()
    return {"total_chunks": collection.count(), "collection_name": settings.COLLECTION_NAME}
