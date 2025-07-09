"""
Main Research Pipeline for processing ArXiv papers and building a vector database.
"""

import os
import json
import re
import ast
from typing import List, Dict, Optional
import logging
import openai

from arxiv_client import ArXivClient
from pdf_processor import PDFProcessor

from chunking import SemanticChunker, ParagraphChunker, TokenChunker
from faiss_database import FaissDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResearchPipeline:
    """Main pipeline for processing research papers and building a vector database."""

    def __init__(
        self,
        downloads_dir: str = "downloads",
        chunks_dir: str = "chunks",
        llm_evaluation_config: Optional[Dict] = None,
        chunking_config: Optional[Dict] = None,
    ):
        """
        Initialize the research pipeline.

        Args:
            downloads_dir: Directory for downloaded PDFs
            chunks_dir: Directory for saved chunks
            llm_evaluation_config: Configuration for LLM evaluation
            chunking_config: Configuration for chunking strategy
        """
        self.downloads_dir = downloads_dir
        self.chunks_dir = chunks_dir
        self.llm_evaluation_config = llm_evaluation_config or {}
        self.chunking_config = chunking_config or {}

        # Create directories
        os.makedirs(self.downloads_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)

        # Initialize components
        self.arxiv_client = ArXivClient()
        self.pdf_processor = PDFProcessor()

        self.chunker = self._get_chunker()
        self.database = FaissDatabase()

        logger.info("Research pipeline initialized")

    def _get_chunker(self):
        """Get the appropriate chunker based on configuration."""
        strategy = self.chunking_config.get("strategy", "semantic")
        model_name = self.chunking_config.get("model", "all-MiniLM-L6-v2")
        
        if strategy == "semantic":
            return SemanticChunker(model_name)
        elif strategy == "paragraph":
            return ParagraphChunker(model_name)
        elif strategy == "token":
            return TokenChunker(model_name)
        else:
            logger.warning(f"Unknown chunking strategy: {strategy}, falling back to semantic")
            return SemanticChunker(model_name)

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


            if not pdf_data["sentences"]:
                logger.warning(f"No sentences extracted from {pdf_path}")
                return None

            # Create chunks using configured strategy
            # Get all config parameters except strategy and model
            chunk_params = {k: v for k, v in self.chunking_config.items() 
                          if k not in ["strategy", "model"]}
            
            chunks = self.chunker.process_sentences(
                pdf_data["sentences"],
                **chunk_params
            )

            # Create chunks using the configured chunker
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

    def deduplicate_chunks(
        self,
        chunks: List[Dict],
        similarity_model=None,
        similarity_threshold: float = 0.95,
        top_k: int = 10,
        keep_strategy: str = "first",
        batch_size: int = 100
    ) -> tuple[List[Dict], Dict]:
        """
        Remove similar chunks using the provided similarity_model's batch hybrid method.
        Args:
            chunks: List of chunks to deduplicate
            similarity_model: Similarity model instance (must have find_topk_hybrid_similarity_batch)
            similarity_threshold: Threshold for considering chunks similar
            top_k: Number of most similar chunks to check per chunk
            keep_strategy: Strategy for keeping chunks ('first', 'longest', 'highest_score')
            batch_size: Batch size for progress logging
        Returns:
            Tuple of (deduplicated_chunks, deduplication_stats)
        """
        logger.info(f"Starting chunk deduplication with {len(chunks)} chunks (hybrid batch)")
        n = len(chunks)
        similar_pairs = []
        # Process in batches for efficiency
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_chunks = chunks[batch_start:batch_end]
            logger.info(f"Processing batch {batch_start+1}-{batch_end} of {n}")
            for i, chunk in enumerate(batch_chunks):
                # Exclude self from candidates
                global_idx = batch_start + i
                candidates = chunks[:global_idx] + chunks[global_idx+1:]
                results = similarity_model.find_topk_hybrid_similarity_batch(
                    queries=[chunk["text"]],
                    candidate_chunks=candidates,
                    top_k_v1=top_k,
                    top_k=top_k,
                    threshold=similarity_threshold,
                    batch_size=batch_size
                )[0]  # Only one query
                for result in results:
                    similar_pairs.append((chunk["chunk_id"], result["chunk_id"], result["score"]))
        # Remove all chunks that appear as similar_chunk_id in any pair
        to_remove = set(pair[1] for pair in similar_pairs)
        dedup_chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in to_remove]
        stats = {
            "original_count": len(chunks),
            "removed_count": len(to_remove),
            "final_count": len(dedup_chunks),
            "reduction_percent": round((len(to_remove) / len(chunks)) * 100, 2),
            "similar_pairs_found": len(similar_pairs),
            "similarity_threshold": similarity_threshold,
            "keep_strategy": keep_strategy,
            "processing_method": "hybrid_batch"
        }
        logger.info(f"Deduplication complete: {stats['removed_count']} chunks removed ({stats['reduction_percent']}% reduction)")
        return dedup_chunks, stats

    def run_complete_pipeline(
        self, 
        query: str, 
        max_results: int = 5,
        deduplicate: bool = True,
        similarity_model=None,
        similarity_threshold: float = 0.95,
        top_k: int = 10,
        keep_strategy: str = "first"
    ) -> Dict:
        """
        Run the complete pipeline with optional steps.
        Args:
            query: Search query
            max_results: Maximum number of papers to process
            deduplicate: Whether to deduplicate chunks (can be skipped)
            similarity_model: Similarity model instance (must have search_documents)
            similarity_threshold: Threshold for deduplication
            top_k: Number of most similar chunks to check per chunk
            keep_strategy: Strategy for keeping chunks
        Returns:
            Pipeline results summary
        """
        logger.info(f"Starting complete pipeline with deduplication for query: {query}")
        # Step 1: Search and download
        papers = self.search_and_download(query, max_results)
        if not papers:
            return {"error": "No papers found"}
        # Step 2: Process papers
        processed_chunks = self.process_papers(papers)
        # Step 3: Deduplicate chunks if requested
        deduplication_stats = None
        if deduplicate and processed_chunks:
            logger.info("Starting chunk deduplication...")
            # Collect all chunks
            all_chunks = []
            for arxiv_id, chunks in processed_chunks.items():
                all_chunks.extend(chunks)
            # Deduplicate
            dedup_chunks, deduplication_stats = self.deduplicate_chunks(
                chunks=all_chunks,
                similarity_model=similarity_model,
                similarity_threshold=similarity_threshold,
                top_k=top_k,
                keep_strategy=keep_strategy
            )
            # Rebuild processed_chunks with deduplicated chunks
            if deduplication_stats['removed_count'] > 0:
                logger.info("Rebuilding database with deduplicated chunks...")
                # Clear existing database
                self.database.clear()
                # Re-add papers with deduplicated chunks
                chunk_by_arxiv = {}
                for chunk in dedup_chunks:
                    arxiv_id = chunk["arxiv_id"]
                    if arxiv_id not in chunk_by_arxiv:
                        chunk_by_arxiv[arxiv_id] = []
                    chunk_by_arxiv[arxiv_id].append(chunk)
                # Update processed_chunks
                processed_chunks = chunk_by_arxiv
                # Re-add to database
                for paper in papers:
                    if paper["arxiv_id"] in processed_chunks:
                        self.add_paper_to_database(paper, processed_chunks[paper["arxiv_id"]])
        # Step 4: Save FAISS database
        if isinstance(self.database, FaissDatabase):
            logger.info("Saving FAISS database to disk...")
            os.makedirs("app/vector_db", exist_ok=True)
            self.database.save("app/vector_db/faiss_index")
            logger.info("FAISS database saved.")
        # Step 5: Get database stats
        db_stats = self.get_database_stats()

        # Step 6: LLM Evaluation (if enabled)
        llm_score = None
        if self.llm_evaluation_config.get("enabled", False):
            logger.info("Starting LLM evaluation...")
            try:
                llm_score = self._evaluate_with_qa_llm_judge(
                    qa_file=self.llm_evaluation_config.get("qa_file"),
                    judge_model=self.llm_evaluation_config.get("judge_model", "gpt-4o"),
                    batch_size=self.llm_evaluation_config.get("batch_size", 2)
                )
                logger.info(f"LLM evaluation completed with score: {llm_score}")
            except Exception as e:
                logger.error(f"LLM evaluation failed: {e}")
        # Summary
        summary = {
            "query": query,
            "papers_found": len(papers),
            "papers_processed": len(processed_chunks),
            "total_chunks": sum(len(chunks) for chunks in processed_chunks.values()),
            "database_stats": db_stats,
            "arxiv_ids": list(processed_chunks.keys()),
            "average_llm_qa_score": llm_score,
        }
        if deduplication_stats:
            summary["deduplication_stats"] = deduplication_stats
        if llm_score is not None:
            summary["llm_evaluation_score"] = llm_score
        logger.info(f"Pipeline with deduplication completed: {summary}")
        return summary


    def _evaluate_with_qa_llm_judge(self, qa_file=None, judge_model="gpt-4o", batch_size=2):
        """
        Evaluate the database using QA pairs and LLM judge.
        
        Args:
            qa_file: Path to QA pairs file (None = use default)
            judge_model: Model to use for judging
            batch_size: Batch size for evaluation
            
        Returns:
            Average score or None if evaluation failed
        """
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
            retrieved = self.search_database(question)
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
