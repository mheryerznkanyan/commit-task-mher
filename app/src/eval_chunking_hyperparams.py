import os
import json
import logging
from tqdm import tqdm
from faiss_database import FaissDatabase
from chunking import SemanticChunker, TokenChunker, ParagraphChunker
from pdf_processor import PDFProcessor
import openai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

QA_FILE = "qa_pairs.json"
DOWNLOADS_DIR = "downloads"
TOP_K = 3
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")  # Can set to gpt-3.5-turbo
MAX_WORKERS = 20  # Increased parallelism
BATCH_SIZE = 5    # Number of QAs per LLM judge call

CHUNKERS = {
    "semantic": SemanticChunker,
    "token": TokenChunker,
    "paragraph": ParagraphChunker,
}
EMBEDDING_MODELS = {
    "MiniLM": "all-MiniLM-L6-v2",
    "SciBERT": "allenai/scibert_scivocab_uncased",
}


def load_qa_pairs():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_paper_texts(arxiv_ids):
    pdf_processor = PDFProcessor()
    paper_texts = {}
    for arxiv_id in set(arxiv_ids):
        pdf_path = os.path.join(DOWNLOADS_DIR, f"{arxiv_id}.pdf")
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            continue
        pdf_data = pdf_processor.process_pdf(pdf_path)
        paper_texts[arxiv_id] = pdf_data["text"]
    return paper_texts

def build_faiss_dbs(paper_texts):
    # Build and cache FAISS DBs for each (arxiv_id, chunker, model) combo
    faiss_dbs = {}
    for chunker_name, chunker_cls in CHUNKERS.items():
        for model_name, model_id in EMBEDDING_MODELS.items():
            for arxiv_id, text in tqdm(paper_texts.items(), desc=f"Chunking {chunker_name}-{model_name}"):
                chunker = chunker_cls(model_name=model_id)
                chunks = chunker.create_chunks(text)
                db = FaissDatabase(model_name=model_id)
                db.add_chunks(chunks)
                faiss_dbs[(arxiv_id, chunker_name, model_name)] = db
    return faiss_dbs

def retrieve_answer_from_db(db, question, top_k=3):
    results = db.search(question, top_k=top_k)
    retrieved_text = " ".join([r["text"] for r in results])
    return retrieved_text

def batch_judge_with_llm(batch, model=JUDGE_MODEL):
    prompt = (
        "You are an expert judge. For each item below, rate the retrieved answer from 1 (irrelevant) to 100 (perfectly answers the question). "
        "Also provide a short justification.\n"
        "Return a JSON list of objects: {'score': int, 'justification': str}.\n"
        "Items:\n"
    )
    for i, qa in enumerate(batch):
        prompt += (
            f"Item {i+1}:\n"
            f"Question: {qa['question']}\n"
            f"Ground-truth answer: {qa['ground_truth']}\n"
            f"Retrieved answer: {qa['retrieved']}\n"
        )
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    import re
    import ast
    import json as pyjson
    try:
        result = pyjson.loads(content)
    except Exception:
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                result = pyjson.loads(match.group(0))
            except Exception:
                result = ast.literal_eval(match.group(0))
        else:
            logger.error("Failed to parse LLM batch judge response.")
            result = []
    return result

def main():
    qa_pairs = load_qa_pairs()
    arxiv_ids = [qa["arxiv_id"] for qa in qa_pairs]
    paper_texts = load_paper_texts(arxiv_ids)
    faiss_dbs = build_faiss_dbs(paper_texts)
    results = {}
    for chunker_name in CHUNKERS:
        for model_name in EMBEDDING_MODELS:
            logger.info(f"Evaluating: Chunker={chunker_name}, Model={model_name}")
            judge_batches = []
            batch = []
            for qa in qa_pairs:
                arxiv_id = qa["arxiv_id"]
                db_key = (arxiv_id, chunker_name, model_name)
                if db_key not in faiss_dbs:
                    continue
                db = faiss_dbs[db_key]
                question = qa["question"]
                ground_truth = qa["answer"]
                retrieved = retrieve_answer_from_db(db, question, top_k=TOP_K)
                batch.append({"question": question, "ground_truth": ground_truth, "retrieved": retrieved})
                if len(batch) == BATCH_SIZE:
                    judge_batches.append(batch)
                    batch = []
            if batch:
                judge_batches.append(batch)
            scores = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {executor.submit(batch_judge_with_llm, b): b for b in judge_batches}
                for future in tqdm(as_completed(future_to_batch), total=len(judge_batches), desc=f"Judging {chunker_name}-{model_name}"):
                    batch_scores = future.result()
                    for item in batch_scores:
                        scores.append(item.get("score", 0))
            avg_score = sum(scores) / len(scores) if scores else 0
            results[(chunker_name, model_name)] = avg_score
            logger.info(f"Avg score for {chunker_name}-{model_name}: {avg_score:.2f}")
    print("\n=== Hyperparameter Tuning Results ===")
    for (chunker_name, model_name), avg_score in results.items():
        print(f"Chunker: {chunker_name:10s} | Model: {model_name:7s} | Avg LLM Score: {avg_score:.2f}")

if __name__ == "__main__":
    main() 