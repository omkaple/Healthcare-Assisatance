"""Quick smoke test for all endpoints."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json

BASE = "http://localhost:8000"
passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} -> {e}")
        failed += 1

# 1. Health
def t_health():
    r = requests.get(f"{BASE}/health")
    d = r.json()
    assert d["status"] == "ok", f"Status: {d['status']}"
    assert d["vector_store"]["total_chunks"] > 0
    print(f"    Chunks: {d['vector_store']['total_chunks']}")

# 2. Greeting
def t_greeting():
    r = requests.post(f"{BASE}/ask", json={"question": "Hi!"})
    d = r.json()
    assert d["workflow"] == "greeting"
    assert "Dr. Aria" in d["answer"] or "hello" in d["answer"].lower() or "Hi" in d["answer"]
    print(f"    {d['answer'][:60]}...")

# 3. RAG Query
def t_rag():
    r = requests.post(f"{BASE}/ask", json={"question": "What is lung cancer?"})
    d = r.json()
    assert d["workflow"] == "rag_pipeline"
    assert d["confidence"] in ["high", "medium", "low"]
    assert len(d["sources"]) > 0
    print(f"    Confidence: {d['confidence']}, Sources: {len(d['sources'])}")

# 4. Appointment
def t_appointment():
    r = requests.post(f"{BASE}/ask", json={"question": "Book appointment for radiation oncology"})
    d = r.json()
    assert d["workflow"] == "appointment_booking_tool"
    assert "Radiation Oncology" in d["answer"]
    print(f"    Slots found in response")

# 5. Clarification
def t_clarification():
    hist = [
        {"role": "user", "content": "What is lung cancer?"},
        {"role": "assistant", "content": "Lung cancer is a type of cancer..."}
    ]
    r = requests.post(f"{BASE}/ask", json={"question": "explain in simple words", "chat_history": hist})
    d = r.json()
    assert d["workflow"] == "clarification"
    print(f"    Re-explained with {len(d['sources'])} sources")

# 6. Thanks
def t_thanks():
    r = requests.post(f"{BASE}/ask", json={"question": "Thank you!"})
    d = r.json()
    assert d["workflow"] == "greeting"
    assert "welcome" in d["answer"].lower()
    print(f"    {d['answer'][:60]}...")

# 7. Static UI
def t_ui():
    r = requests.get(f"{BASE}/")
    assert r.status_code == 200
    assert "Dr. Aria" in r.text
    print(f"    UI loads OK ({len(r.text)} bytes)")

print("=" * 50)
print("HEALTHCARE AI ASSISTANT - SMOKE TESTS")
print("=" * 50)

test("Health Check", t_health)
test("Greeting (Hi!)", t_greeting)
test("RAG Query", t_rag)
test("Appointment Booking", t_appointment)
test("Clarification", t_clarification)
test("Thanks Response", t_thanks)
test("Static UI", t_ui)

print("=" * 50)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed}")
if failed == 0:
    print("ALL TESTS PASSED!")
print("=" * 50)
