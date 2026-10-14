
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = PROJECT_DIR / "vectorstore"
COLLECTION_NAME = "ai_ml_knowledge"


def search_documents(query, k=8):
    """Retrieve the most relevant document chunks for a question."""

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            "Vector store not found. Run build_vectorstore.py first."
        )

    # Use the same embedding model used when building the index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Connect to the existing ChromaDB collection
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    # Retrieve the three most relevant chunks
    results = vectorstore.similarity_search_with_score(query, k=k)

    if not results:
        print("No matching document chunks found.")
        return []

    print(f"\nQuestion: {query}")
    print(f"Retrieved chunks: {len(results)}")

    for index, (document, distance) in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Source: {document.metadata.get('source_file', 'Unknown')}")
        print(f"Page: {document.metadata.get('page', 'Unknown')}")
        print(f"Distance: {distance:.4f}")
        print(f"Text:\n{document.page_content[:700]}")

    return results


if __name__ == "__main__":
    question = input("Ask a question about your PDFs: ").strip()

    if question:
        search_documents(question)
    else:
        print("Please enter a question.")
