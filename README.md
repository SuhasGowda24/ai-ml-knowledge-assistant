# 🤖 DocuMind AI: RAG-Based Document Question Answering System


## 🚀 Live Demo

**Try DocuMind AI:** [Open the deployed application](https://ai-ml-knowledge-assistant-csaaac5cj8sva59rz9mhyp.streamlit.app/)

Ask questions about the uploaded AI and Machine Learning research PDFs and explore source references in the generated answers.


An AI-powered document question-answering application that uses **Retrieval-Augmented Generation (RAG)** to answer questions from research PDFs using retrieved document context.


## 📌 Overview

The AI & Machine Learning Knowledge Assistant is an end-to-end RAG application designed to make technical research documents easier to explore. Users can ask natural-language questions and receive answers generated using relevant passages retrieved from the indexed PDF documents.

The system combines PDF text extraction, text chunking, semantic embeddings, vector search, query expansion, and large language model (LLM) generation in an interactive Streamlit application.

## ✨ Key Features

- **PDF Processing:** Extract text from research PDFs.
- **Text Chunking:** Divide documents into smaller passages for retrieval.
- **Semantic Embeddings:** Represent document chunks as vectors using Sentence Transformers.
- **Vector Search:** Store and retrieve document embeddings using ChromaDB.
- **Query Expansion:** Generate alternative search queries to improve retrieval coverage.
- **Rank Fusion:** Combine ranked retrieval results using Reciprocal Rank Fusion (RRF).
- **Context-Grounded Answers:** Generate responses using retrieved document passages.
- **Source Attribution:** Display retrieved PDF filenames and page numbers.
- **Unsupported-Question Handling:** Acknowledge when the available document context does not support an answer.
- **RAG Evaluation:** Evaluate faithfulness and answer relevancy using RAGAS.
- **Interactive Interface:** Ask questions through a Streamlit web application.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| LLM integration | Groq API |
| LLM | `openai/gpt-oss-20b` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| RAG framework and components | LangChain |
| PDF processing | PyPDF |
| Application interface | Streamlit |
| Evaluation framework | RAGAS |
| Environment configuration | python-dotenv |

## 🏗️ System Architecture

```text
Research PDF Documents
          |
          v
   PDF Text Extraction
          |
          v
     Text Chunking
          |
          v
   Embedding Generation
          |
          v
   ChromaDB Vector Store
          |
          v
     User Question
          |
          v
      Query Expansion
          |
          v
   Similarity Retrieval
          |
          v
 Reciprocal Rank Fusion
          |
          v
   Relevant Context
          |
          v
      Groq LLM
          |
          v
   Generated Answer
          |
          v
 Answer + Retrieved Sources
```

## 📂 Project Structure

```text
PRAICP-Client-Project-5/
│
├── documents/                 # Source PDF documents (local)
├── vectorstore/               # Generated ChromaDB data (local)
│
├── app.py                     # Streamlit application
├── load_documents.py          # PDF loading and text extraction
├── chunk_documents.py         # Text chunking
├── build_vectorstore.py       # Embedding and vector-store creation
├── search_documents.py        # Document retrieval utilities
├── rag_pipeline.py            # RAG question-answering pipeline
├── evaluate_rag.py            # RAGAS evaluation
├── evaluation_dataset.csv     # Evaluation questions and references
├── evaluation_results.json    # Generated evaluation results (local)
├── show_evaluation.py         # Evaluation result display utility
├── debug_retrieval.py         # Retrieval debugging and comparison
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git exclusions
├── .env                       # API credentials (not committed)
└── README.md
```

*Note: The structure above describes the project files. Only files actually present in the repository should be listed as committed files. The `documents/`, `vectorstore/`, `.env`, and generated evaluation output may be excluded from Git.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/SuhasGowda24/ai-ml-knowledge-assistant.git
cd ai-ml-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key. Never publish your actual API key or commit the `.env` file to GitHub.

5. Source Documents

The documents/ directory contains the research PDFs used as the knowledge source for DocuMind AI.

If the PDFs are included in the repository, they will be available after cloning. Otherwise, place your permitted PDF documents in the documents/ directory before building the knowledge base.

6. Build the Vector Database

python build_vectorstore.py

This extracts and processes the PDF content, generates embeddings, and builds the local ChromaDB vector store.

### 7. Launch the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, typically:

```text
http://localhost:8501
```

You can now ask questions about the indexed research documents.

## 🧪 Evaluation

The project uses RAGAS to evaluate answer quality with two metrics.

- **Faithfulness:** Measures whether the generated answer is supported by the retrieved context.
- **Answer Relevancy:** Measures how relevant the generated answer is to the user's question.

### Preliminary Evaluation Results

| Metric | Score |
|---|---:|
| Average Faithfulness | 0.9471 |
| Average Answer Relevancy | 0.7257 |
| Evaluation Questions | 10 |

**Evaluation note:** These are preliminary results, not a fully controlled single-model evaluation. The first eight questions were evaluated using `openai/gpt-oss-120b`, while the final two were evaluated using `openai/gpt-oss-20b` after API quota limitations interrupted the original run.

For a more consistent comparison, rerun the complete evaluation dataset using the same generation and evaluation configuration.

### Run the evaluation

After configuring the API credentials and preparing the knowledge base, run:

```bash
python evaluate_rag.py
```

The evaluation requires valid provider credentials and is subject to model availability, API quotas, and rate limits.

## 🔍 Testing and Validation

The application was manually tested with questions covering:

- Document chunking and retrieval accuracy.
- Embeddings and semantic similarity search.
- The role of vector databases in a RAG pipeline.
- Questions outside the scope of the uploaded documents.
- Retrieval of source filenames and page metadata.

The application successfully generated relevant answers for tested concepts and displayed retrieved sources. It also returned an appropriate fallback response when asked about quantum computing, a topic not covered by the supplied PDFs.

These tests demonstrate functional behavior but do not guarantee that every generated answer or source attribution is correct.

## 🧩 Challenges Faced

- **Document preprocessing:** Preparing extracted PDF text for effective retrieval.
- **Chunking strategy:** Balancing context preservation and retrieval precision.
- **Retrieval quality:** Improving the relevance of retrieved passages using query expansion and rank fusion.
- **Grounded generation:** Encouraging the LLM to answer from the supplied context rather than unsupported assumptions.
- **Evaluation constraints:** Working with API quotas and maintaining consistent evaluation configurations.
- **Source attribution:** Presenting retrieved PDF names and page metadata to help users verify answers.

## ⚠️ Limitations

- Answer quality depends on the quality, coverage, and structure of the indexed documents.
- Relevant passages may not always be retrieved, even when information exists in the source PDFs.
- The model may produce inaccurate statements or imperfect source attribution.
- Questions outside the knowledge base cannot be answered reliably from the supplied documents.
- Evaluation results depend on the dataset, models, prompts, and configuration used.
- Groq API availability, rate limits, and usage quotas can interrupt generation or evaluation.
- The current application focuses on PDF-based question answering rather than a general-purpose AI assistant.

## 🚀 Future Improvements

- Evaluate all questions using a consistent model configuration.
- Improve retrieval using hybrid search and reranking.
- Add retrieval metrics such as Hit Rate, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG).
- Expand the evaluation dataset with more representative questions.
- Improve source attribution and page-level citation verification.
- Add support for additional document formats.
- Add automated tests for retrieval and answer quality.
- Deploy the application with secure API-key management.

## 🎯 Conclusion

The AI & Machine Learning Knowledge Assistant demonstrates an end-to-end Retrieval-Augmented Generation workflow for research-document question answering. It integrates PDF processing, text chunking, semantic embeddings, ChromaDB, retrieval optimization, and LLM-based response generation through a Streamlit interface.

The project also incorporates RAGAS evaluation and source display to support the assessment and verification of generated answers. Functional testing demonstrated that the application can answer questions about its indexed documents and acknowledge when the supplied context does not contain sufficient information.

This project provides a foundation for further experimentation with retrieval optimization, evaluation consistency, and production-ready document intelligence systems.

