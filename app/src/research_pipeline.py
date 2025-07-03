"""
Main Research Pipeline for processing ArXiv papers and building a vector database.
"""

import os
import json
from typing import List, Dict, Optional
import logging
import openai

from arxiv_client import ArXivClient
from pdf_processor import PDFProcessor
from chunking.paragraph_chunker import ParagraphChunker
from faiss_database import FaissDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResearchPipeline:
    """Main pipeline for processing research papers and building a vector database."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.downloads_dir = cfg.data.paths.downloads_dir
        self.chunks_dir = cfg.data.paths.chunks_dir
        self.model_name = cfg.model.embedding.model_name

        # Create directories
        os.makedirs(self.downloads_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)

        # Initialize components
        self.arxiv_client = ArXivClient()
        self.pdf_processor = PDFProcessor()
        self.chunker = ParagraphChunker(model_name=self.model_name)
        self.database = FaissDatabase(model_name=self.model_name)

        logger.info(f"Research pipeline initialized with model: {self.model_name}")

    def search_and_download(
        self, query: str, max_results: int = 5, max_workers: int = 8
    ) -> List[Dict]:
        """
        Search for papers and download them.

        Args:
            query: Search query
            max_results: Maximum number of papers to download
            max_workers: Number of parallel download threads

        Returns:
            List of paper information dictionaries
        """
        # Search for papers
        papers = self.arxiv_client.search(query, max_results)

        if not papers:
            logger.warning("No papers found for query")
            return []

        # Download PDFs
        arxiv_ids = [paper["arxiv_id"] for paper in papers]
        downloaded = self.arxiv_client.download_papers(
            arxiv_ids, self.downloads_dir, max_workers=max_workers
        )

        # Add PDF paths to paper info
        for paper in papers:
            paper["pdf_path"] = downloaded.get(paper["arxiv_id"])

        logger.info(f"Downloaded {len(downloaded)} papers")
        return papers

    def process_paper(self, pdf_path: str, arxiv_id: str) -> Optional[List[Dict]]:
        """
        Process a single paper: extract text, create chunks, and add to database.

        Args:
            pdf_path: Path to PDF file
            arxiv_id: ArXiv ID of the paper

        Returns:
            List of chunks or None if failed
        """
        try:
            pdf_data = self.pdf_processor.process_pdf(pdf_path)
            text = pdf_data["text"]
            sentences = pdf_data["sentences"]

            # Debug: print first 500 chars and number of blank lines
            logger.debug(f"First 500 chars of text for {arxiv_id}: {text[:500]}")
            logger.debug(f"Number of blank lines (\\n\\n) in text: {text.count('\n\n')}")

            # If only one paragraph, fallback to sentence-based chunking
            if text.count('\n\n') < 2:
                logger.warning(f"Only one paragraph found in {pdf_path}, falling back to sentence-based chunking.")
                chunk_size = 5
                chunks = []
                for i in range(0, len(sentences), chunk_size):
                    chunk_text = " ".join(sentences[i:i+chunk_size])
                    chunk_embedding = self.chunker.get_embedding(chunk_text)
                    chunk = {
                        "text": chunk_text,
                        "embedding": chunk_embedding,
                        "chunk_id": i // chunk_size,
                        "arxiv_id": arxiv_id,
                    }
                    chunks.append(chunk)
            else:
                chunks = self.chunker.create_chunks(text)
                for chunk in chunks:
                    chunk["arxiv_id"] = arxiv_id

            logger.info(f"Processed paper {arxiv_id}: {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error processing paper {arxiv_id}: {e}")
            return None

    def save_chunks(self, chunks: List[Dict], arxiv_id: str) -> str:
        """
        Save chunks to JSON file.

        Args:
            chunks: List of chunks
            arxiv_id: ArXiv ID

        Returns:
            Path to saved file
        """
        file_path = os.path.join(self.chunks_dir, f"{arxiv_id}_chunks.json")

        # Convert numpy arrays to lists for JSON serialization
        serializable_chunks = []
        for chunk in chunks:
            serializable_chunk = chunk.copy()
            serializable_chunk["embedding"] = chunk["embedding"].tolist()
            serializable_chunks.append(serializable_chunk)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved chunks to {file_path}")
        return file_path

    def add_paper_to_database(self, paper: Dict, chunks: List[Dict]) -> bool:
        """
        Add a paper and its chunks to the vector database.

        Args:
            paper: Paper information dictionary
            chunks: List of chunks

        Returns:
            True if successful
        """
        return self.database.add_paper(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            summary=paper["summary"],
            link=paper["link"],
            chunks=chunks,
        )

    def process_papers(self, papers: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Process multiple papers.

        Args:
            papers: List of paper information dictionaries

        Returns:
            Dictionary mapping ArXiv IDs to chunks
        """
        all_chunks = {}

        for paper in papers:
            if not paper.get("pdf_path"):
                logger.warning(f"No PDF path for paper {paper['arxiv_id']}")
                continue

            chunks = self.process_paper(paper["pdf_path"], paper["arxiv_id"])
            if chunks:
                # Save chunks
                self.save_chunks(chunks, paper["arxiv_id"])

                # Add to database
                success = self.add_paper_to_database(paper, chunks)
                if success:
                    all_chunks[paper["arxiv_id"]] = chunks
                else:
                    logger.error(f"Failed to add paper {paper['arxiv_id']} to database")

        logger.info(f"Processed {len(all_chunks)} papers successfully")
        return all_chunks

    def search_database(
        self, query: str, top_k: int = 10, filter_arxiv_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search the vector database.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_arxiv_id: Optional filter by ArXiv ID

        Returns:
            List of search results
        """
        return self.database.search(query, top_k, filter_arxiv_id)

    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        return self.database.get_collection_info()

    def _evaluate_with_qa_llm_judge(self, qa_file=None, top_k=3, judge_model="gpt-4o", batch_size=2):
        import re, ast
        # Always look for qa_pairs.json in app/qa_pairs.json relative to project root
        if qa_file is None:
            qa_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../qa_pairs.json'))
        # Load QA pairs
        if not os.path.exists(qa_file):
            logger.warning(f"QA file not found: {qa_file}")
            return None
        with open(qa_file, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        if not qa_pairs:
            logger.warning("No QA pairs loaded.")
            return None
        # Prepare batches for LLM judge
        batches = []
        batch = []
        for qa in qa_pairs:
            question = qa["question"]
            ground_truth = qa["answer"]
            # Retrieve answer from DB (optionally filter by arxiv_id)
            retrieved = self.search_database(question, top_k=top_k)
            retrieved_text = " ".join([r["text"] for r in retrieved])
            batch.append({"question": question, "ground_truth": ground_truth, "retrieved": retrieved_text})
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        # LLM judge
        openai.api_key = os.getenv("OPENAI_API_KEY")
        all_scores = []
        for batch in batches:
            prompt = (
                f"You are an expert judge. For each item below, rate the retrieved answer from 1 (irrelevant) to 100 (perfectly answers the question). "
                f"Also provide a short justification.\n"
                f"Return a JSON list of objects: {{'score': int, 'justification': str}}.\n"
                f"Items:\n"
            )
            for i, qa in enumerate(batch):
                prompt += (
                    f"Item {i+1}:\n"
                    f"Question: {qa['question']}\n"
                    f"Ground-truth answer: {qa['ground_truth']}\n"
                    f"Retrieved answer: {qa['retrieved']}\n"
                )
            response = openai.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
            except Exception:
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group(0))
                    except Exception:
                        result = ast.literal_eval(match.group(0))
                else:
                    logger.error("Failed to parse LLM batch judge response.")
                    result = []
            for item in result:
                all_scores.append(item.get("score", 0))
                logger.info(f"LLM Judge: Score={item.get('score', 0)}, Justification={item.get('justification', '')}")
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        logger.info(f"Average LLM QA Score: {avg_score:.2f} over {len(all_scores)} QA pairs.")
        return avg_score

    def run_complete_pipeline(self, query: str, max_results: int = 5) -> Dict:
        """
        Run the complete pipeline: search, download, process, and index.

        Args:
            query: Search query
            max_results: Maximum number of papers to process

        Returns:
            Pipeline results summary
        """
        logger.info(f"Starting complete pipeline for query: {query}")

        # Step 1: Search and download
        papers = self.search_and_download(query, max_results)

        if not papers:
            return {"error": "No papers found"}

        # Step 2: Process papers
        processed_chunks = self.process_papers(papers)

        # Automatically save FAISS database after processing
        if isinstance(self.database, FaissDatabase):
            logger.info("Saving FAISS database to disk...")
            os.makedirs("app/vector_db", exist_ok=True)
            self.database.save("app/vector_db/faiss_index")
            logger.info("FAISS database saved.")

        # Step 3: Get database stats
        db_stats = self.get_database_stats()

        # Step 4: Run QA/LLM evaluation and log results
        avg_score = self._evaluate_with_qa_llm_judge()
        if avg_score is not None:
            logger.info(f"Average LLM QA Score: {avg_score:.2f}")

        # Summary
        summary = {
            "query": query,
            "papers_found": len(papers),
            "papers_processed": len(processed_chunks),
            "total_chunks": sum(len(chunks) for chunks in processed_chunks.values()),
            "database_stats": db_stats,
            "arxiv_ids": list(processed_chunks.keys()),
            "average_llm_qa_score": avg_score,
        }

        logger.info(f"Pipeline completed: {summary}")
        return summary
# Average LLM QA Score: 32.55 over 100 QA pairs.