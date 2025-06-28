import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from research_pipeline import ResearchPipeline
from faiss_database import FaissDatabase
from complex_qa import (
    decompose_question,
    retrieve_answers,
    compose_final_answer,
    load_faiss_db,
)

FAISS_PATH = "app/vector_db/faiss_index"
CHUNKS_DIR = "app/chunks"
DOWNLOADS_DIR = "app/downloads"

app = FastAPI()


class CreateDBRequest(BaseModel):
    query: str = Field("transformers in healthcare", min_length=1)
    num_papers: int = Field(5, ge=1, le=100)

    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query string must not be empty.")
        return v


class RelatedTextsRequest(BaseModel):
    query_text: str = Field("transformers in healthcare", min_length=1)
    top_k: int = Field(5, ge=1, le=50)

    @validator("query_text")
    def query_text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query text must not be empty.")
        return v


class ComplexQARequest(BaseModel):
    question: str = Field(
        "How are transformer models used in healthcare?", min_length=1
    )
    top_k: int = Field(3, ge=1, le=20)

    @validator("question")
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question must not be empty.")
        return v


@app.post("/create_faiss_db")
def create_faiss_db(req: CreateDBRequest):
    """
    Create a FAISS database from arXiv papers matching the query.
    """
    try:
        pipeline = ResearchPipeline(downloads_dir=DOWNLOADS_DIR, chunks_dir=CHUNKS_DIR)
        results = pipeline.run_complete_pipeline(req.query, req.num_papers)
        return {"status": "created", "summary": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faiss_exists")
def faiss_exists():
    """
    Check if the FAISS database exists on disk.
    """
    try:
        exists = os.path.exists(FAISS_PATH + ".index") and os.path.exists(
            FAISS_PATH + ".chunks.pkl"
        )
        return {"exists": exists}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/empty_faiss_db")
def empty_faiss_db():
    """
    Delete the FAISS database files from disk. Returns is_empty: true if DB is empty, false if not.
    """
    try:
        for ext in [".index", ".chunks.pkl"]:
            path = FAISS_PATH + ext
            if os.path.exists(path):
                os.remove(path)
        is_empty = not (
            os.path.exists(FAISS_PATH + ".index")
            or os.path.exists(FAISS_PATH + ".chunks.pkl")
        )
        return {"is_empty": is_empty}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_related_texts")
def get_related_texts_api(req: RelatedTextsRequest):
    """
    Get related texts from the FAISS database for a given query.
    """
    try:
        db = FaissDatabase()
        db.load(FAISS_PATH)
        results = db.search(req.query_text, top_k=req.top_k)
        return [{"text": r["text"], "score": r["score"]} for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_complex_qa")
def complex_qa_api(req: ComplexQARequest):
    """
    Answer a complex question using the FAISS database and RAG pipeline.
    """
    try:
        db = load_faiss_db(FAISS_PATH)
        sub_questions = decompose_question(req.question)
        sub_answers = retrieve_answers(sub_questions, db, top_k=req.top_k)
        final_answer = compose_final_answer(req.question, sub_answers)
        return {
            "question": req.question,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
            "final_answer": final_answer,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
