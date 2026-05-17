# Healthcare AI Assistant 🏥

A RAG-based AI assistant for cancer-related healthcare information, built with FastAPI, ChromaDB, and Google Gemini.

## Features

- **Document Ingestion**: Reads `.txt` and `.pdf` files, splits into chunks, generates embeddings, and stores in ChromaDB.
- **RAG Question Answering**: Retrieves relevant context and generates grounded answers using Google Gemini LLM.
- **Agentic Workflow**: Routes appointment-related queries to a mock booking tool; knowledge queries to the RAG pipeline.
- **Source Citations**: Every answer includes source document references.
- **Confidence Scoring**: Responses include high/medium/low confidence levels.
- **Anti-Hallucination**: The system prompt enforces answers only from provided context.
- **API with Swagger Docs**: FastAPI with interactive docs at `/docs`.
- **Docker Support**: Dockerfile and docker-compose for containerized deployment.

## Project Structure

```
healthcare-ai-assistant/
├── app/
│   ├── __init__.py
│   ├── config.py        # Settings from .env
│   ├── embeddings.py    # Sentence-transformers embeddings
│   ├── llm.py           # LLM integration (Gemini/Ollama/fallback)
│   ├── rag.py           # Document ingestion & retrieval (ChromaDB)
│   ├── agent.py         # Agentic routing & prompt engineering
│   └── main.py          # FastAPI endpoints
├── data/                # Healthcare documents (cancer-focused)
├── vector_store/        # ChromaDB persistence (auto-created)
├── tests/
│   └── test_main.py
├── .env                 # Configuration (add your API key here)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key

Edit `.env` and add your Google Gemini API key (free at https://aistudio.google.com/app/apikey):

```
GEMINI_API_KEY=your_api_key_here
```

### 3. Run the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest
```

### 5. Ask Questions

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the treatment options for breast cancer?"}'
```

## API Endpoints

| Method | Endpoint  | Description                          |
|--------|-----------|--------------------------------------|
| GET    | /health   | System health check                  |
| POST   | /ingest   | Ingest documents into vector store   |
| POST   | /ask      | Ask a healthcare question            |
| GET    | /docs     | Interactive Swagger API docs         |

## System Prompt

```
You are a professional healthcare AI assistant specializing in cancer-related information.
You MUST follow these rules strictly:

1. ONLY answer based on the provided CONTEXT. Do NOT use external knowledge.
2. If the answer is NOT found in the context, respond: "I could not find this information in the provided documents."
3. NEVER guess, speculate, or make up information.
4. NEVER provide direct medical diagnoses or unsafe medical advice.
5. Always recommend consulting a qualified healthcare professional.
6. Keep responses clear, professional, and empathetic.
7. Mention the source document for transparency.
8. Use bullet points or numbered lists for clarity.
```

## Docker Deployment

```bash
docker-compose up --build
```

## Technologies

- **FastAPI** - API framework
- **ChromaDB** - Vector database
- **Sentence-Transformers** - Local embeddings (all-MiniLM-L6-v2)
- **Google Gemini** - LLM for answer generation
- **Pydantic** - Data validation
