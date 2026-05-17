"""Tests for the Healthcare AI Assistant API."""

# pyrefly: ignore [missing-import]
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test the /health endpoint returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_store" in data


def test_ask_missing_question():
    """Test /ask with missing question returns 422."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_short_question():
    """Test /ask with too-short question returns 422."""
    response = client.post("/ask", json={"question": "ab"})
    assert response.status_code == 422


def test_ask_appointment_booking():
    """Test appointment booking routing via /ask endpoint."""
    response = client.post(
        "/ask",
        json={"question": "Can I book a cardiology appointment for Monday?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["workflow"] == "appointment_booking_tool"
    assert "available" in data["answer"].lower() or "slot" in data["answer"].lower() or "scheduling" in data["answer"].lower()
    assert data["confidence"] == "high"


def test_ask_rag_empty_store():
    """Test RAG query when vector store is empty returns graceful message."""
    response = client.post(
        "/ask",
        json={"question": "What are the symptoms of lung cancer?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["workflow"] == "rag_pipeline"
    assert "answer" in data


def test_ingest_endpoint():
    """Test the /ingest endpoint processes documents."""
    response = client.post("/ingest")
    assert response.status_code == 200
    data = response.json()
    assert "chunks_ingested" in data
    assert data["chunks_ingested"] > 0


def test_ask_after_ingest():
    """Test RAG query after ingestion returns relevant answer."""
    # Make sure documents are ingested
    client.post("/ingest")

    response = client.post(
        "/ask",
        json={"question": "What is breast cancer?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["workflow"] == "rag_pipeline"
    assert len(data["sources"]) > 0
    assert data["confidence"] in ["high", "medium", "low"]
