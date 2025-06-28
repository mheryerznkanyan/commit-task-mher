import os
import openai
import logging
from dotenv import load_dotenv
from faiss_database import FaissDatabase

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")


def decompose_question(question, model="gpt-4o", temperature=0.3, max_tokens=150):
    prompt = (
        "Break down the following complex question into clear, answerable sub-questions. "
        "List each sub-question on a new line:\n"
        f"Question: {question}\n"
        "Sub-questions:"
    )
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    sub_questions = [
        q.strip("- ").strip()
        for q in response.choices[0].message.content.split("\n")
        if q.strip()
    ]
    return [q for q in sub_questions if q]


def retrieve_answers(sub_questions, db, top_k=3):
    answers = []
    for sub_q in sub_questions:
        results = db.search(sub_q, top_k=top_k)
        context = " ".join([r["text"] for r in results])
        logger.debug(
            f"[DEBUG] Retrieved context for sub-question '{sub_q}':\n{context[:300]}...\n"
        )
        answers.append((sub_q, context))
    return answers


def compose_final_answer(
    question, sub_answers, model="gpt-4o", temperature=0.1, max_tokens=300
):
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a in sub_answers])
    logger.debug(
        "[DEBUG] Context passed to LLM for final answer composition:\n"
        + context[:1000]
        + "\n"
    )
    prompt = (
        f"Given the following sub-answers, compose a comprehensive answer to the original question. "
        f"Original question: {question}\n"
        f"{context}\n"
        "Final answer:"
    )
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def load_faiss_db(path="app/vector_db/faiss_index"):
    db = FaissDatabase()
    if os.path.exists(path + ".index") and os.path.exists(path + ".chunks.pkl"):
        db.load(path)
        logger.info(f"Loaded FAISS index from {path}.")
    else:
        logger.info("Using in-memory FAISS DB (from current session)")
    return db
