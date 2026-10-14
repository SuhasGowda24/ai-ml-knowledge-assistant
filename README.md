# AI & Machine Learning Knowledge Assistant

An AI-powered knowledge assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from PDF documents using relevant retrieved context.

## Overview

This project demonstrates an end-to-end RAG pipeline that processes PDF documents, splits their content into chunks, generates embeddings, stores them in a vector database, and retrieves relevant information to generate context-grounded answers.

## Key Features

- PDF document loading and text extraction
- Text chunking for efficient retrieval
- Semantic search using vector embeddings
- Vector storage and similarity-based retrieval
- Context-aware question answering
- RAG evaluation using Faithfulness and Answer Relevancy metrics
- Interactive application interface

## Technology Stack

- **Language:** Python
- **LLM Integration:** Groq API
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Evaluation:** RAGAS
- **Application:** Streamlit
- **Document Processing:** PyPDF

## Project Structure

```text
PRAICP-Client-Project-5/
├── documents/                 # Source PDF documents
├── app.py                     # Application interface
├── load_documents.py          # PDF loading and extraction
├── chunk_documents.py         # Text chunking
├── build_vectorstore.py       # Vector database creation
├── search_documents.py        # Semantic retrieval
├── rag_pipeline.py            # RAG question-answering pipeline
├── evaluate_rag.py            # RAG evaluation
├── evaluation_dataset.csv      # Evaluation questions and references
├── requirements.txt            # Python dependencies
├── .gitignore                 # Files excluded from Git
└── README.md
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/SuhasGowda24/ai-ml-knowledge-assistant.git
cd ai-ml-knowledge-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your actual API key to GitHub.

### 5. Prepare the vector database

After placing your permitted source PDFs in the `documents/` folder, run:

```bash
python build_vectorstore.py
```

### 6. Launch the application

```bash
streamlit run app.py
```

## Evaluation

The project uses RAGAS to evaluate answer quality through:

- **Faithfulness:** Measures whether the generated answer is supported by the retrieved context.
- **Answer Relevancy:** Measures how relevant the generated answer is to the user's question.

Run the evaluation script using:

```bash
python evaluate_rag.py
```

Evaluation requires valid API credentials and may incur usage limits depending on the selected provider and model.

## Limitations

- Answer quality depends on document quality and retrieval performance.
- Responses may contain errors even when generated with retrieved context.
- Evaluation results depend on the selected model, evaluation dataset, and available API quota.
- Processing large document collections may require additional optimization.

## Future Improvements

- Support additional document formats.
- Improve chunking and retrieval strategies.
- Add retrieval metrics such as Hit Rate, MRR, and NDCG.
- Expand the evaluation dataset.
- Deploy the application for broader access.


