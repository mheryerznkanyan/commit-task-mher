import os
import json
import logging
from arxiv_client import ArXivClient
from pdf_processor import PDFProcessor
import openai
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

SEARCH_QUERY = "transformers in healthcare"  # Change as needed
NUM_PAPERS = 5  # Change as needed
NUM_QA_PER_PAPER = 20  # Fixed number of Q&A pairs per paper
DOWNLOADS_DIR = "downloads"
QA_OUTPUT_FILE = "qa_pairs.json"


def generate_qa_pairs(text, num_pairs=20, model="gpt-4o"):
    prompt = (
        f"Given the following research paper content, generate {num_pairs} question-answer pairs. "
        "Each question should be answerable from the text. Return the result as a JSON list of objects with 'question' and 'answer' fields.\n"
        "Content:\n"
        f"{text[:12000]}"  # Truncate to fit context window
    )
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    try:
        qa_pairs = json.loads(content)
    except Exception:
        # Try to extract JSON from the response
        import re
        import ast
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                qa_pairs = json.loads(match.group(0))
            except Exception:
                qa_pairs = ast.literal_eval(match.group(0))
        else:
            logger.error("Failed to parse Q&A pairs from OpenAI response.")
            qa_pairs = []
    return qa_pairs


def main():
    arxiv = ArXivClient()
    pdf_processor = PDFProcessor()

    logger.info(f"Searching for {NUM_PAPERS} papers with query: '{SEARCH_QUERY}'")
    papers = arxiv.search(SEARCH_QUERY, max_results=NUM_PAPERS)
    arxiv_ids = [paper["arxiv_id"] for paper in papers]
    logger.info(f"Downloading {len(arxiv_ids)} papers...")
    downloaded = arxiv.download_papers(arxiv_ids, save_dir=DOWNLOADS_DIR)

    if not downloaded:
        logger.error("No papers downloaded. Exiting.")
        return

    qa_pairs = []
    for arxiv_id, pdf_path in downloaded.items():
        logger.info(f"Processing PDF: {pdf_path}")
        pdf_data = pdf_processor.process_pdf(pdf_path)
        logger.info(f"Generating {NUM_QA_PER_PAPER} Q&A pairs for paper {arxiv_id} using OpenAI...")
        paper_qa_pairs = generate_qa_pairs(pdf_data["text"], num_pairs=NUM_QA_PER_PAPER)
        for qa in paper_qa_pairs:
            qa["arxiv_id"] = arxiv_id
        qa_pairs.extend(paper_qa_pairs)

    with open(QA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {QA_OUTPUT_FILE}")

if __name__ == "__main__":
    main() 