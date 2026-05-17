"""
Agent module with basic agentic/tool-based workflow.
Routes questions to either RAG pipeline or mock appointment booking tool.
Includes prompt engineering for safe, grounded, conversational healthcare responses.
Supports conversation history for natural follow-up questions.
"""

import re
import logging
import random
from datetime import datetime, timedelta

from app.rag import retrieve_relevant_chunks
from app.llm import get_llm_response, transform_query, clean_rag_context
from app.config import settings

logger = logging.getLogger(__name__)

# ─── System Prompt for RAG-based QA (Conversational) ───
SYSTEM_PROMPT = """You are Dr. Aria — a warm, empathetic, and knowledgeable AI healthcare assistant 
specializing in cancer care. You work at a leading cancer care center and you genuinely care 
about every patient who reaches out to you.

YOUR PERSONALITY:
- You are warm, kind, and approachable — like a trusted doctor friend.
- You speak in a clear, conversational tone — never robotic or overly formal.
- You use simple language that anyone can understand, avoiding unnecessary jargon.
- If someone says they don't understand, you patiently re-explain using everyday words and analogies.
- You show empathy — acknowledge concerns and emotions before diving into information.
- You use encouraging language: "Great question!", "I'm glad you asked!", "That's very important to know."

YOUR RULES:
1. ONLY answer based on the provided CONTEXT below. Do NOT use external knowledge.
2. If the context doesn't contain the answer, say: "I don't have that specific information in my records right now, but I'd recommend asking your doctor about this."
3. NEVER guess, speculate, or make up medical facts.
4. NEVER provide direct diagnoses or tell someone they have a disease.
5. Always gently suggest consulting their healthcare provider for personalized advice.
6. Use bullet points, short paragraphs, and emojis sparingly (like 💡, ✅, 📋) to make answers easy to scan.
7. When citing information, weave it naturally — don't just dump raw text.

{history_section}

CONTEXT FROM HEALTHCARE DOCUMENTS:
{context}

PATIENT'S QUESTION: {question}

Respond as Dr. Aria — be warm, clear, and helpful. If they asked you to simplify or re-explain, 
use simpler words and analogies:"""

# ─── Greeting responses for non-medical messages ───
GREETINGS = {
    "keywords": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy", "hii", "hiii"],
    "responses": [
        "Hello! 👋 I'm Dr. Aria, your cancer care AI assistant. I'm here to help you with any questions about cancer — types, treatments, prevention, screening, or even scheduling an appointment. What would you like to know?",
        "Hi there! 😊 I'm Dr. Aria, and I'm here to help you with cancer-related questions. Whether it's about symptoms, treatments, or general wellness — feel free to ask me anything!",
        "Hey! Welcome! I'm Dr. Aria, your friendly healthcare AI assistant. I specialize in cancer care information. How can I help you today?",
    ],
}

THANKS_KEYWORDS = ["thank", "thanks", "thank you", "thx", "appreciate"]
UNCLEAR_KEYWORDS = ["don't understand", "dont understand", "didn't understand", "not clear", 
                     "confused", "explain again", "simplify", "simple words", "easy words",
                     "what do you mean", "in simple", "too complex", "too complicated", 
                     "eli5", "layman", "plain english"]

# ─── Appointment keywords for routing ───
APPOINTMENT_KEYWORDS = [
    "book", "schedule", "appointment", "slot", "available",
    "booking", "reschedule", "cancel appointment",
]

DEPARTMENTS = [
    "Medical Oncology", "Surgical Oncology", "Radiation Oncology",
    "Hematology/Oncology", "Oncology Genetics", "Palliative Care",
    "Oncology Nutrition", "Psycho-Oncology", "Cardiology", "General",
]


def _is_greeting(question: str) -> bool:
    """Check if the message is a greeting."""
    q_lower = question.lower().strip()
    # Exact or near-exact match
    if q_lower in GREETINGS["keywords"] or q_lower.rstrip("!.") in GREETINGS["keywords"]:
        return True
    # Starts with greeting
    for kw in GREETINGS["keywords"]:
        if q_lower.startswith(kw + " ") or q_lower.startswith(kw + ","):
            return True
    return False


def _is_clarification_request(question: str) -> bool:
    """Check if user is asking for simpler explanation."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in UNCLEAR_KEYWORDS)


def _is_thanks(question: str) -> bool:
    """Check if user is saying thanks."""
    q_lower = question.lower().strip()
    return any(kw in q_lower for kw in THANKS_KEYWORDS)


def _is_appointment_query(question: str) -> bool:
    """Check if the question is about appointment booking."""
    q_lower = question.lower()
    matches = sum(1 for kw in APPOINTMENT_KEYWORDS if kw in q_lower)
    return matches >= 1


def _extract_department(question: str) -> str:
    """Extract department from the question, default to General."""
    q_lower = question.lower()
    for dept in DEPARTMENTS:
        if dept.lower() in q_lower:
            return dept
    if "cardio" in q_lower:
        return "Cardiology"
    if "surg" in q_lower:
        return "Surgical Oncology"
    if "radiat" in q_lower:
        return "Radiation Oncology"
    if "palliati" in q_lower:
        return "Palliative Care"
    if "nutri" in q_lower:
        return "Oncology Nutrition"
    if "genetic" in q_lower:
        return "Oncology Genetics"
    if "oncolog" in q_lower:
        return "Medical Oncology"
    return "Medical Oncology"


def _extract_date(question: str) -> str:
    """Extract date reference from question."""
    q_lower = question.lower()
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days:
        if day in q_lower:
            return day.capitalize()
    if "today" in q_lower:
        return "Today"
    if "tomorrow" in q_lower:
        return "Tomorrow"
    return "Next available"


def check_available_slots(department: str, date: str) -> dict:
    """
    Mock tool: Check available appointment slots.
    Returns simulated available slots for demonstration.
    """
    logger.info(f"[TOOL] check_available_slots(department='{department}', date='{date}')")

    base_times = ["9:00 AM", "10:30 AM", "11:00 AM", "1:00 PM", "2:30 PM", "3:00 PM", "4:00 PM"]
    num_slots = random.randint(2, 4)
    available = random.sample(base_times, min(num_slots, len(base_times)))
    available.sort()

    return {
        "department": department,
        "date": date,
        "available_slots": available,
        "doctor": f"Dr. {'Smith' if random.random() > 0.5 else 'Johnson'}",
        "location": "Cancer Care Center, Building A",
        "notes": "Please arrive 15 minutes early. Bring your insurance card and photo ID.",
    }


def _calculate_confidence(chunks: list[dict]) -> str:
    """Calculate confidence level based on retrieval distances."""
    if not chunks:
        return "low"
    avg_distance = sum(c["distance"] for c in chunks) / len(chunks)
    similarity = 1 - (avg_distance / 2)
    if similarity >= settings.CONFIDENCE_THRESHOLD_HIGH:
        return "high"
    elif similarity >= settings.CONFIDENCE_THRESHOLD_MEDIUM:
        return "medium"
    return "low"


def _build_history_section(chat_history: list[dict]) -> str:
    """Build conversation history section for the prompt."""
    if not chat_history:
        return ""

    # Keep last 6 exchanges to stay within token limits
    recent = chat_history[-6:]
    lines = ["PREVIOUS CONVERSATION (for context — patient may refer to earlier topics):"]
    for msg in recent:
        role = "Patient" if msg.get("role") == "user" else "Dr. Aria"
        text = msg.get("content", "")[:300]  # Truncate long messages
        lines.append(f"  {role}: {text}")

    return "\n".join(lines)


def handle_query(question: str, chat_history: list[dict] = None) -> dict:
    """
    Main entry point: route the question to the appropriate handler.

    Agentic workflow:
    1. Greetings / thanks → friendly conversational response
    2. Appointment booking → mock scheduling tool
    3. Clarification requests → re-retrieve + ask LLM to simplify
    4. Knowledge questions → RAG pipeline

    Args:
        question: The user's question string.
        chat_history: List of {"role": "user"/"assistant", "content": "..."} dicts.

    Returns:
        Dict with 'answer', 'sources', 'confidence', and 'workflow' keys.
    """
    logger.info(f"Processing query: '{question}'")
    if chat_history is None:
        chat_history = []

    # ─── Route 0: Greetings ───
    if _is_greeting(question):
        logger.info("Routing to greeting response.")
        return {
            "answer": random.choice(GREETINGS["responses"]),
            "sources": [],
            "confidence": "high",
            "workflow": "greeting",
        }

    # ─── Route 0b: Thanks ───
    if _is_thanks(question):
        return {
            "answer": "You're very welcome! 😊 I'm always here if you have more questions. "
                      "Remember — no question is too small when it comes to your health. Take care! 💙",
            "sources": [],
            "confidence": "high",
            "workflow": "greeting",
        }

    # ─── Route 1: Appointment booking (agentic tool) ───
    if _is_appointment_query(question):
        logger.info("Routing to appointment booking tool.")
        department = _extract_department(question)
        date = _extract_date(question)
        slots = check_available_slots(department, date)

        slot_list = ", ".join(slots["available_slots"])
        answer = (
            f"Of course! I'd be happy to help you schedule an appointment. 😊\n\n"
            f"Here are the available slots for **{slots['department']}** on **{slots['date']}**:\n\n"
            f"🕐 Available times: {slot_list}\n"
            f"👨‍⚕️ Doctor: {slots['doctor']}\n"
            f"📍 Location: {slots['location']}\n\n"
            f"📋 {slots['notes']}\n\n"
            f"To confirm, just call our scheduling line at **(555) 123-4567**. "
            f"Is there anything else I can help with?"
        )

        return {
            "answer": answer,
            "sources": [{"document": "appointment_scheduling_policy.txt", "chunk": "Mock appointment tool"}],
            "confidence": "high",
            "workflow": "appointment_booking_tool",
        }

    # ─── Route 2: Clarification (re-explain simpler) ───
    if _is_clarification_request(question) and chat_history:
        logger.info("Routing to clarification handler.")
        # Find the last bot response and the ORIGINAL question (skip clarification requests)
        last_bot_msg = ""
        last_user_question = ""
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant" and not last_bot_msg:
                last_bot_msg = msg.get("content", "")
            elif msg.get("role") == "user" and not last_user_question:
                content = msg.get("content", "").strip()
                # Skip the current clarification request itself
                if content.lower() != question.strip().lower() and not _is_clarification_request(content):
                    last_user_question = content
            if last_bot_msg and last_user_question:
                break

        # Re-retrieve context for the original topic
        search_query = last_user_question or question
        logger.info(f"Re-searching for original topic: '{search_query}'")
        chunks = retrieve_relevant_chunks(search_query)

        if chunks:
            context = "\n\n---\n\n".join(
                f"[Source: {c['document']}]\n{c['chunk']}" for c in chunks
            )
            simplify_prompt = f"""You are Dr. Aria, a warm and caring cancer care AI assistant.

The patient previously asked about a topic and you gave them an answer, but they said they 
didn't understand. Please RE-EXPLAIN the same topic using:
- Very simple, everyday language (as if explaining to a 10-year-old)
- Short sentences
- Real-world analogies where helpful
- Bullet points for clarity

YOUR PREVIOUS ANSWER:
{last_bot_msg[:800]}

REFERENCE DOCUMENTS:
{context}

PATIENT SAYS: {question}

Re-explain in simpler words, be encouraging and patient:"""

            try:
                answer = get_llm_response(simplify_prompt)
            except Exception:
                # Friendly simplified fallback
                clean = re.sub(r"\[Source:.*?\]\s*", "", chunks[0]['chunk']).strip()
                answer = (
                    f"No worries! Let me explain that in simpler terms 😊\n\n"
                    f"Here's the main point:\n\n"
                    f"• {clean[:400]}\n\n"
                    f"In simple words — this is about what doctors have found in their research. "
                    f"If you want me to explain any specific part, just ask! I'm here to help. 💙"
                )

            sources = [{"document": c["document"], "chunk": c["chunk"][:200] + "..."} for c in chunks]
            return {
                "answer": answer,
                "sources": sources,
                "confidence": _calculate_confidence(chunks),
                "workflow": "clarification",
            }

    # ─── Route 3: RAG-based knowledge retrieval ───
    logger.info("Routing to RAG pipeline.")
    
    # Rewrite question using chat history for better context search
    standalone_question = transform_query(question, chat_history)
    if standalone_question != question:
        logger.info(f"Query transformed for search: '{question}' -> '{standalone_question}'")
        
    chunks = retrieve_relevant_chunks(standalone_question)

    if not chunks:
        return {
            "answer": "Hmm, I couldn't find information about that in my healthcare documents. 🤔\n\n"
                      "Could you try rephrasing your question? Or if you'd like, I can help with:\n"
                      "- Cancer types, symptoms, and treatments\n"
                      "- Prevention and screening guidelines\n"
                      "- Appointment scheduling\n"
                      "- Telehealth consultations\n\n"
                      "Just ask! I'm here to help. 😊",
            "sources": [],
            "confidence": "low",
            "workflow": "rag_pipeline",
        }

    # Build context from retrieved chunks
    raw_context = "\n\n---\n\n".join(
        f"[Source: {c['document']}]\n{c['chunk']}" for c in chunks
    )

    # Check spelling and grammar of RAG context
    logger.info("Checking RAG context for spelling and grammatical mistakes...")
    context = clean_rag_context(raw_context)

    # Call LLM
    try:
        answer = get_llm_response(
            question=question,
            context=context,
            chat_history=chat_history
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        from app.llm import _fallback_response
        answer = _fallback_response(question, context)

    # Build sources list
    sources = [
        {"document": c["document"], "chunk": c["chunk"][:200] + "..."}
        for c in chunks
    ]

    confidence = _calculate_confidence(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "workflow": "rag_pipeline",
    }
