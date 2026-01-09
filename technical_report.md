# Technical Report: Backstory Consistency Checker
## Kharagpur Data Science Hackathon 2026 - Track A (NLP/GenAI)

### 1. System Architecture

**Pipeline Overview:**
1. Novel Loading & Preprocessing (.txt/.pdf)
2. Text Chunking (2000 chars with 200 overlap)
3. Pathway Vector Indexing with OpenAI embeddings
4. Backstory Claim Extraction
5. Semantic Retrieval using vector similarity
6. LLM-based Consistency Analysis with Causal Reasoning
7. Binary Decision Logic
8. Output Generation

### 2. Pathway Framework Implementation

**Why Pathway Chose:**
- Real-time data processing capabilities
- Built-in vector store for semantic search
- Scalable for 100k+ word novels
- Seamless OpenAI integration

**Key Pathway Components Used:**
```python
1. pw.debug.table_from_pandas() - Data ingestion
2. OpenAIEmbedder() - Text embeddings
3. Vector similarity search - Retrieval logic