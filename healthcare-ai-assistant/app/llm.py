"""
LLM integration module.
Uses Google Gemini with system_instruction and proper chat history via types.Content.
Falls back to template-based responses when the API is unavailable.
"""

import re
import logging
from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level GenAI client (cached)
_client = None


def _get_client() -> genai.Client:
    """Returns a cached GenAI client."""
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your .env file. "
                "Get a free API key from https://aistudio.google.com/app/apikey"
            )
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logger.info("Google GenAI client initialized.")
    return _client


def get_llm_response(
    question: str,
    context: str,
    chat_history: list[dict] = None,
    system_prompt: str = None,
) -> str:
    """
    Call Gemini with system_instruction, chat history, and context.

    Args:
        question: The user's current question.
        context: RAG-retrieved document context.
        chat_history: List of {"role": "user"/"assistant", "content": "..."} dicts.
        system_prompt: Optional custom system instruction override.

    Returns:
        The LLM response text, or a fallback response on failure.
    """
    if not system_prompt:
        system_prompt = _default_system_prompt(context)
    else:
        # Inject context into custom system prompt if it has a placeholder
        if "{context}" in system_prompt:
            system_prompt = system_prompt.replace("{context}", context)

    # Build chat history as types.Content objects
    contents = []
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 6 messages for context
            role = "user" if msg.get("role") == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.get("content", "")[:500])]
                )
            )

    # Append the current user question
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)]
        )
    )

    # Try Gemini API
    try:
        client = _get_client()
        logger.info(f"Sending to Gemini model: {settings.GEMINI_MODEL}")

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                http_options=types.HttpOptions(timeout=10_000),
            ),
        )

        logger.info("Received response from Gemini.")
        return response.text

    except Exception as e:
        logger.warning(f"Gemini unavailable ({type(e).__name__}). Using fallback.")
        return _fallback_response(question, context)


def _default_system_prompt(context: str) -> str:
    """Build the default healthcare system instruction with context."""
    return f"""You are Dr. Aria — a warm, empathetic, and knowledgeable AI healthcare assistant 
specializing in cancer care. You genuinely care about every patient who reaches out.

YOUR PERSONALITY:
- Warm, kind, and approachable — like a trusted doctor friend.
- Clear, conversational tone — never robotic or overly formal.
- Use simple language that anyone can understand, avoid unnecessary jargon.
- If someone says they don't understand, patiently re-explain using everyday words and analogies.
- Show empathy — acknowledge concerns and emotions before diving into information.
- Use encouraging phrases: "Great question!", "I'm glad you asked!", "That's very important to know."

YOUR RULES:
1. ONLY answer based on the provided CONTEXT below. Do NOT use external knowledge.
2. If the context doesn't contain the answer, say: "I don't have that specific information in my records right now, but I'd recommend asking your doctor about this."
3. NEVER guess, speculate, or make up medical facts.
4. NEVER provide direct diagnoses or tell someone they have a disease.
5. Always gently suggest consulting their healthcare provider for personalized advice.
6. Use bullet points, short paragraphs, and emojis sparingly (like 💡, ✅, 📋) to make answers easy to scan.
7. CRITICAL: The context comes from PDFs, which often have spelling mistakes, broken words, or messy tables (like raw numbers and body parts). You MUST fix all spelling and grammatical errors before outputting.
8. If the context contains a messy table of cancer data, convert it into a neat, easy-to-read list of cancer types. NEVER output raw, unformatted tabular OCR text.
9. When citing information, weave it naturally — don't just dump raw text.

CONTEXT FROM HEALTHCARE DOCUMENTS:
{context}

Respond as Dr. Aria — be warm, clear, and helpful:"""


def _fallback_response(question: str, context: str) -> str:
    """
    Conversational fallback when LLM is unavailable.
    Formats the retrieved context in Dr. Aria's friendly voice.
    Cleans up messy OCR tables using regex.
    """
    logger.info("Using template-based fallback response.")

    if not context or context.strip() == "":
        return (
            "I appreciate you reaching out! Unfortunately, I couldn't find specific information "
            "about that in my healthcare documents. Could you try asking in a different way? "
            "I'm here to help! 😊"
        )

    # Clean up — remove [Source: ...] tags and dividers
    clean = re.sub(r"\[Source:.*?\]\s*", "", context).strip()
    clean = re.sub(r"\n---\n", "\n\n", clean).strip()
    
    # Remove messy OCR tables (lines that are mostly numbers or single words followed by many numbers)
    clean = re.sub(r"^(?:[\w\s&]+)?(?:\s*\d{1,3}(?:,\d{3})*)+\s*$", "", clean, flags=re.MULTILINE)
    
    # Clean up excessive newlines left behind
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

    # Truncate to reasonable length
    if len(clean) > 800:
        clean = clean[:800] + "..."

    return (
        f"I'm experiencing high traffic right now, but here's what I found in our records:\n\n"
        f"{clean}\n\n"
        f"💡 For a better answer, please try asking again in a few minutes!"
    )


def get_simplified_response(
    question: str,
    context: str,
    previous_answer: str,
    chat_history: list[dict] = None,
) -> str:
    """
    Re-explain a previous answer in simpler terms.
    Used when the user says 'I don't understand'.
    """
    system_prompt = f"""You are Dr. Aria, a warm and caring cancer care AI assistant.

The patient previously asked about a topic and you gave them an answer, but they said they 
didn't understand. Please RE-EXPLAIN the same topic using:
- Very simple, everyday language (as if explaining to a 10-year-old)
- Short sentences
- Real-world analogies where helpful
- Bullet points for clarity
- Be encouraging and patient

YOUR PREVIOUS ANSWER:
{previous_answer[:800]}

REFERENCE DOCUMENTS:
{context}

Re-explain in simpler words:"""

    try:
        client = _get_client()
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=question)]
            )
        ]

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                http_options=types.HttpOptions(timeout=10_000),
            ),
        )
        return response.text

    except Exception as e:
        logger.warning(f"Gemini unavailable for simplification ({type(e).__name__}). Using fallback.")
        clean = re.sub(r"\[Source:.*?\]\s*", "", context).strip()
        clean = re.sub(r"\n---\n", "\n\n", clean).strip()
        if len(clean) > 800:
            clean = clean[:800] + "..."

        return (
            f"No worries! Let me explain that in simpler terms 😊\n\n"
            f"Here's the key info:\n\n"
            f"{clean}\n\n"
            f"💡 If any specific part is still unclear, just point it out and I'll explain further!"
        )


def transform_query(question: str, chat_history: list[dict] = None) -> str:
    """
    Takes a follow-up question, leverages the existing chat history to 
    rewrite it into a standalone question, and prevents the raw question 
    from staying in the history.
    """
    if not chat_history:
        return question
        
    try:
        client = _get_client()
        
        # Build history
        contents = []
        for msg in chat_history[-6:]:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.get("content", "")[:500])]
                )
            )
            
        # 1. Temporarily append the user's raw question to the history
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=question)]
            )
        )
        
        # 2. Call the Gemini API to rewrite the query
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a query rewriting expert. Based on the provided chat history, "
                    "rephrase the 'Follow Up user Question' into a complete, standalone question "
                    "that can be understood without the chat history.\n"
                    "Only output the rewritten question and nothing else."
                ),
                http_options=types.HttpOptions(timeout=10_000),
            ),
        )
        
        # (3. and 4. The raw question is not saved permanently as contents is a local list)
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Query transformation failed: {e}. Using original question.")
        return question


def clean_rag_context(context: str) -> str:
    """
    Uses the LLM to check and fix spelling, grammatical mistakes, and messy OCR tables 
    from the raw vector database output before it is used.
    """
    if not context.strip():
        return context
        
    system_prompt = (
        "You are an expert medical text editor. The user will provide text extracted "
        "from a vector database (PDF OCR text). Your task is to ONLY fix spelling mistakes, "
        "grammatical errors, and format any messy tables into clean lists. "
        "Do NOT answer any questions or add new information. Just output the corrected text."
    )
    
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=context)])],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                http_options=types.HttpOptions(timeout=10_000),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Grammar correction failed: {e}. Using raw context.")
        return context
