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
import logging
from dotenv import load_dotenv
from src.complex_qa import (
    decompose_question,
    retrieve_answers,
    compose_final_answer,
    load_faiss_db,
)
import hydra
from omegaconf import DictConfig

load_dotenv()
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


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


@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    db = load_faiss_db(cfg.paths.faiss_index)
    logger.info("\nComplex Question Answering (type 'exit' to quit)")
    while True:
        question = input("\nAsk your complex question: ").strip()
        if question.lower() in ("exit", "quit"):
            logger.info("Exiting.")
            break
        if not question:
            continue
        logger.info("\nDecomposing question...")
        sub_questions = decompose_question(
            question,
            model=cfg.openai.model,
            temperature=cfg.openai.temperature,
            max_tokens=cfg.openai.max_tokens,
        )
        logger.info("Sub-questions:")
        for i, sq in enumerate(sub_questions, 1):
            logger.info(f"  {i}. {sq}")
        logger.info("\nRetrieving context for each sub-question...")
        sub_answers = retrieve_answers(sub_questions, db, top_k=cfg.db.top_k)
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
