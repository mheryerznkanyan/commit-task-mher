"""
Complex Question Answering over Vector Database

- Decomposes a complex question into sub-questions using OpenAI
- Retrieves relevant context for each sub-question from FAISS
- Composes a final answer using OpenAI

Set your OpenAI API key in the environment variable OPENAI_API_KEY.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import openai
from faiss_database import FaissDatabase # type: ignore
import logging
from dotenv import load_dotenv
from complex_qa import decompose_question, retrieve_answers, compose_final_answer, load_faiss_db

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


def decompose_question(question):
    prompt = (
        "Break down the following complex question into clear, answerable sub-questions. "
        "List each sub-question on a new line:\n"
        f"Question: {question}\n"
        "Sub-questions:"
    )
    response = openai.chat.completions.create(
        model="gpt-4o",  # or "gpt-4-1106-preview", "gpt-4-turbo", etc.
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.3,
    )
    sub_questions = [q.strip("- ").strip() for q in response.choices[0].message.content.split("\n") if q.strip()]
    return [q for q in sub_questions if q]


def retrieve_answers(sub_questions, db, top_k=3):
    answers = []
    for sub_q in sub_questions:
        results = db.search(sub_q, top_k=top_k)
        context = " ".join([r['text'] for r in results])
        logger.debug(f"[DEBUG] Retrieved context for sub-question '{sub_q}':\n{context[:300]}...\n")
        answers.append((sub_q, context))
    return answers


def compose_final_answer(question, sub_answers):
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a in sub_answers])
    prompts = [
        (
            "Given the following sub-answers, compose a comprehensive answer to the original question. "
            f"Original question: {question}\n"
            f"{context}\n"
            "Final answer:"
        ),
        (
            "Using only the information in the sub-answers below, write a clear and concise answer to the original question. "
            f"Original question: {question}\n"
            f"{context}\n"
            "Answer:"
        ),
        (
            "Based on the following context, provide a detailed and well-structured answer to the user's question. "
            f"Question: {question}\n"
            f"{context}\n"
            "Your answer:"
        ),
        (
            "Summarize the following sub-answers into a single, informative response to the original question. "
            f"Original question: {question}\n"
            f"{context}\n"
            "Summary answer:"
        ),
    ]
    answers = []
    for i, prompt in enumerate(prompts, 1):
        logger.debug(f"\n[DEBUG] Trying prompt {i} for final answer composition...\n")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
        answers.append((i, prompt.splitlines()[-1], answer))
    return answers


def judge_answers(question, answers):
    prompt = (
        f"Question: {question}\n"
        "Here are several answers from different algorithms/prompts:\n"
    )
    for i, ans in enumerate(answers, 1):
        prompt += f"Answer {i}: {ans}\n"
    prompt += (
        "Please rate each answer from 1 (poor) to 5 (excellent) for helpfulness and accuracy. "
        "Then, explain which answer is best and why."
    )
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1,
    )
    logger.info(response.choices[0].message.content)


def main():
    db = load_faiss_db()
    logger.info("\nComplex Question Answering (type 'exit' to quit)")
    while True:
        question = input("\nAsk your complex question: ").strip()
        if question.lower() in ("exit", "quit"):
            logger.info("Exiting.")
            break
        if not question:
            continue
        logger.info("\nDecomposing question...")
        sub_questions = decompose_question(question)
        logger.info("Sub-questions:")
        for i, sq in enumerate(sub_questions, 1):
            logger.info(f"  {i}. {sq}")
        logger.info("\nRetrieving context for each sub-question...")
        sub_answers = retrieve_answers(sub_questions, db)
        for i, (sq, ans) in enumerate(sub_answers, 1):
            logger.debug(f"\nSub-question {i}: {sq}\nRelevant context: {ans[:200]}...")
        logger.info("\nComposing final answer...")
        final_answer = compose_final_answer(question, sub_answers)
        logger.info("\nFinal Answer:\n%s", final_answer)
        logger.info("\n=== SUMMARY ===")
        logger.info(f"QUESTION: {question}")
        logger.info(f"ANSWER: {final_answer}")

if __name__ == "__main__":
    main() 