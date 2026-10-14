
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_documents import load_pdf_documents


def split_documents(documents):
    """Split loaded documents into smaller overlapping chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total pages loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nSample chunk:")
        print(chunks[0].page_content[:500])
        print("\nSample chunk metadata:")
        print(chunks[0].metadata)

    return chunks


if __name__ == "__main__":
    documents = load_pdf_documents()

    if documents:
        chunks = split_documents(documents)
