
from rag_pipeline import create_rag_components, retrieve_with_query_expansion

vectorstore, llm = create_rag_components()

question = "What are embeddings in a RAG system?"
k = 8


def display_results(title, results):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"Chunks retrieved: {len(results)}")
    print("=" * 70)

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\nResult {rank} | Distance: {distance:.4f}")
        print(f"Source: {document.metadata.get('source_file', 'Unknown')}")
        print(f"Page metadata: {document.metadata.get('page', 'Unknown')}")
        print(document.page_content[:500])
        print("-" * 50)


# Test 1: Standard similarity search
standard_results = vectorstore.similarity_search_with_score(
    question,
    k=k,
)

display_results("STANDARD SIMILARITY SEARCH", standard_results)

# Test 2: Query expansion + Reciprocal Rank Fusion
rrf_results = retrieve_with_query_expansion(
    question,
    vectorstore,
    llm,
    k=k,
)

display_results("QUERY EXPANSION + RRF", rrf_results)
