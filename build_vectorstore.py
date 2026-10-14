
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from chunk_documents import split_documents
from load_documents import load_pdf_documents


PROJECT_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"
COLLECTION_NAME = "ai_ml_knowledge"


def build_vectorstore():
    # Step 1: Load PDFs
    documents = load_pdf_documents()

    if not documents:
        print("No documents loaded. Add PDFs to the documents folder.")
        return

    # Step 2: Split the documents into chunks
    chunks = split_documents(documents)

    if not chunks:
        print("No text chunks were created.")
        return

    # Step 3: Initialize the local embedding model
    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Step 4: Create and persist the ChromaDB vector store
    print("\nCreating ChromaDB vector store...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )

    print(f"\nVector store saved at: {VECTORSTORE_DIR}")
    print(f"Chunks indexed: {len(chunks)}")

    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
