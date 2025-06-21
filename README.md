# Research Paper Processing Pipeline

A comprehensive system for searching, downloading, processing, and indexing ArXiv research papers using semantic chunking and vector search.

## Features

- **ArXiv Integration**: Search and download papers from ArXiv
- **PDF Processing**: Extract and preprocess text from PDF files
- **Semantic Chunking**: Create meaningful text chunks using sentence transformers
- **Vector Database**: Store and search chunks using Qdrant or FAISS vector database
- **Modular Architecture**: Clean, object-oriented design with separate components

## Architecture

The system is built with a modular, object-oriented architecture:

- `ArXivClient`: Handles ArXiv API interactions
- `PDFProcessor`: Extracts and preprocesses PDF text
- `SemanticChunker`: Creates semantic chunks with embeddings
- `QdrantDatabase`: Manages Qdrant vector storage and retrieval
- `FaissDatabase`: In-memory FAISS vector search (no server required)
- `ResearchPipeline`: Orchestrates the complete workflow

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For Qdrant (optional, for persistent vector DB):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

FAISS works in-memory and does not require a server.

## Usage

### Basic Usage (Qdrant)

```python
from research_pipeline import ResearchPipeline

pipeline = ResearchPipeline()
results = pipeline.run_complete_pipeline("large language models", max_results=5)
search_results = pipeline.search_database("transformer architecture", top_k=10)
```

### Using FAISS Vector Database

```python
from faiss_database import FaissDatabase

# Create FAISS DB
faiss_db = FaissDatabase()

# Add chunks (from your chunking pipeline)
faiss_db.add_chunks(chunks)

# Search
results = faiss_db.search("ex", top_k=5)

# Save/load index
faiss_db.save("faiss_index")
faiss_db.load("faiss_index")
```

### Component Usage

```python
# ArXiv client
from arxiv_client import ArXivClient
client = ArXivClient()
papers = client.search("machine learning", max_results=10)
client.download_papers([paper['arxiv_id'] for paper in papers])

# PDF processing
from pdf_processor import PDFProcessor
processor = PDFProcessor()
pdf_data = processor.process_pdf("path/to/paper.pdf")

# Semantic chunking
from semantic_chunker import SemanticChunker
chunker = SemanticChunker()
chunks = chunker.process_sentences(pdf_data['sentences'])

# Vector database (Qdrant or FAISS)
from qdrant_database import QdrantDatabase
db = QdrantDatabase()
db.add_paper("arxiv_id", "title", "summary", "link", chunks)
results = db.search("query", top_k=5)

from faiss_database import FaissDatabase
faiss_db = FaissDatabase()
faiss_db.add_chunks(chunks)
results = faiss_db.search("query", top_k=5)
```

### Running the Demo

```bash
python main.py
```

## Configuration

### Qdrant Setup

1. Install Docker
2. Run Qdrant container:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Model Configuration

The system uses `all-MiniLM-L6-v2` by default for sentence embeddings. You can change this in the `SemanticChunker`, `QdrantDatabase`, or `FaissDatabase` classes.

## File Structure

```
├── arxiv_client.py      # ArXiv API client
├── pdf_processor.py     # PDF text extraction
├── semantic_chunker.py  # Semantic chunking
├── qdrant_database.py   # Qdrant vector database
├── faiss_database.py    # FAISS vector database
├── research_pipeline.py # Main pipeline
├── main.py             # Demo script
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## API Reference

### FaissDatabase

In-memory vector search using FAISS. No server required.

#### Methods
- `add_chunks(chunks)`: Add chunk dictionaries to the index
- `search(query, top_k)`: Search for similar chunks
- `save(path)`: Save index and metadata
- `load(path)`: Load index and metadata
- `clear()`: Clear all data

### ResearchPipeline

Main class that orchestrates the complete workflow.

#### Methods
- `run_complete_pipeline(query, max_results)`: Run the complete pipeline
- `search_and_download(query, max_results)`: Search and download papers
- `process_papers(papers)`: Process multiple papers
- `search_database(query, top_k, filter_arxiv_id)`: Search the vector database
- `get_database_stats()`: Get database statistics

### ArXivClient

Handles ArXiv API interactions.

#### Methods
- `search(query, max_results)`: Search for papers
- `download_pdf(arxiv_id, save_dir)`: Download a single PDF
- `download_papers(arxiv_ids, save_dir)`: Download multiple PDFs

### PDFProcessor

Extracts and preprocesses PDF text.

#### Methods
- `extract_text(pdf_path)`: Extract raw text from PDF
- `preprocess_text(text)`: Clean and normalize text
- `split_sentences(text)`: Split text into sentences
- `process_pdf(pdf_path)`: Complete PDF processing pipeline

### SemanticChunker

Creates semantic chunks from sentences.

#### Methods
- `create_chunks(sentences, chunk_size, overlap)`: Create initial chunks
- `merge_similar_chunks(chunks, similarity_threshold)`: Merge similar chunks
- `process_sentences(sentences, chunk_size, overlap, similarity_threshold)`: Complete chunking pipeline

### QdrantDatabase

Manages vector storage and retrieval.

#### Methods
- `add_paper(arxiv_id, title, summary, link, chunks)`: Add paper to database
- `search(query, top_k, filter_arxiv_id)`: Search for similar chunks
- `get_collection_info()`: Get database statistics
- `delete_paper(arxiv_id)`: Delete paper from database
- `clear_collection()`: Clear all data

## Performance

- **Text Extraction**: Uses PyPDF2 for reliable PDF text extraction
- **Embeddings**: Sentence transformers for high-quality semantic embeddings
- **Vector Search**: Qdrant for persistent, FAISS for in-memory fast similarity search
- **Chunking**: Configurable chunk size and overlap for optimal results

## Error Handling

The system includes comprehensive error handling and logging:
- Network errors during downloads
- PDF processing failures
- Database connection issues
- Invalid input validation

## Logging

The system uses Python's logging module with configurable levels:
- INFO: General progress information
- WARNING: Non-critical issues
- ERROR: Critical failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License. 