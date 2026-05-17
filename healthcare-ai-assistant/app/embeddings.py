"""
Embedding model module.
Uses Google Gemini's text-embedding-004 model for document vectorization and query embedding.
Falls back to sentence-transformers if Gemini embeddings are unavailable.
"""

import logging
from app.config import settings

logger = logging.getLogger(__name__)

# Module-level cache for the embedding client
_genai_client = None


def _get_genai_client():
    """Returns a cached Google GenAI client."""
    global _genai_client
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logger.info("Google GenAI embedding client initialized.")
    return _genai_client


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Google's text-embedding-004 model.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.
    """
    try:
        client = _get_genai_client()
        model = "text-embedding-004"

        # Google API supports batch embedding
        # Process in batches of 100 (API limit)
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Embedding batch {i // batch_size + 1} ({len(batch)} texts) with {model}")

            result = client.models.embed_content(
                model=model,
                contents=batch,
            )

            # Extract embedding vectors from the response
            for embedding in result.embeddings:
                all_embeddings.append(embedding.values)

        logger.info(f"Generated {len(all_embeddings)} embeddings using {model}.")
        return all_embeddings

    except Exception as e:
        logger.warning(f"Google embedding failed: {e}. Falling back to sentence-transformers.")
        return _fallback_embeddings(texts)


def _fallback_embeddings(texts: list[str]) -> list[list[float]]:
    """Fallback to local sentence-transformers embeddings."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()
