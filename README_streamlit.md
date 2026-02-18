# Streamlit Graph-RAG Demo

This UI demonstrates your Graph-RAG system with transparency:
- Shows the **raw KG-retrieved context** (nodes + relationships)
- Shows the **Cypher query executed**
- Shows the **final LLM answer**
- Optionally shows **embedding retrieval output**

## Setup

Create/activate your venv, then install dependencies:

```bash
pip install -r requirements.txt
```

## Required environment variables

Set these in your shell (or in a `.env` you load yourself):

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `OPENROUTER_API_KEY` (optional, only needed for LLM answers)

If you don’t set Neo4j env vars, the code will try to read `M3/config.txt`.

## Run

From the repo root:

```bash
python -m streamlit run app.py
```

## Notes

- Retrieval modes:
  - **Cypher (KG baseline)**: uses `M3/pipeline.process_query` and displays the returned subgraph.
  - **Embeddings** / **Hybrid**: uses `M3/embeddings.semantic_search` (requires embeddings stored in Neo4j and Neo4j GDS for cosine similarity).
