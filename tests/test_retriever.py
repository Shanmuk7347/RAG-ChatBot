from retrievers.bm25 import get_bm25_retriever

bm25 = get_bm25_retriever("test", 2)

for query in ["arrays", "Two pointers", "Sliding window", "Prefix Sum"]:
    results = bm25.invoke(query)

    for i, doc in enumerate(results, start=1):
        print(f"Result for {query}:{i}:")
        print("="*50)
        print(doc.metadata)
        print(doc.page_content)