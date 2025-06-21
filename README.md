# arXiv Paper Search & QA API

A FastAPI application for searching arXiv, downloading papers, extracting text, semantic chunking, and answering complex questions using FAISS vector search.

## Quickstart (Docker)

1. **Build the Docker image:**
   ```bash
   docker build -t arxiv-fastapi-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 arxiv-fastapi-app
   ```

3. **Open the API docs:**
   [http://localhost:8000/docs](http://localhost:8000/docs)

Use the interactive docs to test endpoints for creating the FAISS DB, searching, and complex QA.

---

## Running Tests

To run the test suite (requires pytest):

```bash
pytest app/tests/test_faiss_and_api.py -s
```

This will run all core functionality and API tests, printing average similarity scores and sample answers.

---

For development or testing, see the `app/tests/` directory for example tests.

---

## Future Work

- **Semantic chunking improvements:**
  - Explore more advanced or adaptive chunking strategies for better context retention and retrieval.

- **Scalability:**
  - Design for distributed processing  GPU availability and storage to handle large-scale paper collections and queries.

- **Handling complex data:**
  - Support extraction and semantic search for images, graphs, tables, and formulas in PDFs.
  - Evaluate and compare PyMuPDF vs [unstructured.io](https://unstructured.io/) for richer document parsing.

- **Crash detection and monitoring:**
  - Integrate syslog or similar logging for crash/error detection and alerting.

- **Caching:**
  - Add Redis or similar caching layer to speed up repeated queries and reduce load.

- **Vector database selection:**
  - Evaluate and allow switching between FAISS, Qdrant, and other vector DBs based on use case (local, cloud, scalability, etc). 