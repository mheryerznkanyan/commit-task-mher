import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from api import app, FAISS_PATH
from faiss_database import FaissDatabase
from complex_qa import load_faiss_db, decompose_question, retrieve_answers, compose_final_answer

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_faiss_db():
    # Ensure FAISS DB exists for tests
    if not (os.path.exists(FAISS_PATH + ".index") and os.path.exists(FAISS_PATH + ".chunks.pkl")):
        # Create DB via API
        resp = client.post("/create_faiss_db", json={"query": "transformers in healthcare", "num_papers": 3})
        assert resp.status_code == 200
    yield
    # Teardown: clean up DB
    for ext in [".index", ".chunks.pkl"]:
        path = FAISS_PATH + ext
        if os.path.exists(path):
            os.remove(path)


def test_faiss_text_extraction_and_score(setup_faiss_db):
    db = FaissDatabase()
    db.load(FAISS_PATH)
    results = db.search("transformers in healthcare", top_k=5)
    assert results, "No results from FAISS search!"
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"Average similarity score: {avg_score:.3f}")
    assert avg_score > 0.1  # Arbitrary threshold for test


def test_api_get_related_texts(setup_faiss_db):
    resp = client.post("/get_related_texts", json={"query_text": "transformers in healthcare", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert all("text" in d and "score" in d for d in data)


def test_api_faiss_exists(setup_faiss_db):
    resp = client.get("/faiss_exists")
    assert resp.status_code == 200
    assert resp.json()["exists"] is True


def test_api_empty_faiss_db():
    # Empty the DB
    resp = client.post("/empty_faiss_db")
    assert resp.status_code == 200
    assert resp.json()["is_empty"] is True
    # Check again
    resp2 = client.get("/faiss_exists")
    assert resp2.status_code == 200
    assert resp2.json()["exists"] is False


def test_complex_qa_judgement(setup_faiss_db):
    # Use the pipeline to answer a question
    db = load_faiss_db(FAISS_PATH)
    question = "How are transformer models used in healthcare?"
    sub_questions = decompose_question(question)
    sub_answers = retrieve_answers(sub_questions, db, top_k=2)
    final_answer = compose_final_answer(question, sub_answers)
    # Use a simple heuristic: answer should mention 'transformer' and 'health'
    assert "transformer" in final_answer.lower()
    assert "health" in final_answer.lower()
    print(f"Final answer: {final_answer}")


def test_api_get_complex_qa(setup_faiss_db):
    resp = client.post("/get_complex_qa", json={"question": "How are transformer models used in healthcare?", "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "final_answer" in data
    assert "transformer" in data["final_answer"].lower()
    assert "health" in data["final_answer"].lower() 