
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"


def load_pdf_documents():
    """Load all PDF files from the documents folder."""

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {DOCUMENTS_DIR}")
        return []

    all_documents = []

    for pdf_path in pdf_files:
        print(f"\nLoading: {pdf_path.name}")

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Add the source filename to each page's metadata
            for page in pages:
                page.metadata["source_file"] = pdf_path.name

            all_documents.extend(pages)

            print(f"Pages loaded: {len(pages)}")

        except Exception as error:
            print(f"Could not load {pdf_path.name}: {error}")

    print(f"\nTotal PDF files found: {len(pdf_files)}")
    print(f"Total pages loaded: {len(all_documents)}")

    return all_documents


if __name__ == "__main__":
    documents = load_pdf_documents()

    if documents:
        print("\nSample extracted text:")
        print(documents[0].page_content[:1000])
