from retrievers.hybrid import get_hybrid_retriever

chat_id = "test"

hybrid = get_hybrid_retriever(chat_id)
queries = [
    "arrays",
    "Prefix Sum",
    "When to do evaluation of LLMs ?",
]


for query in queries:
    docs = hybrid.invoke(query)
    print(f"=============={query}=================")
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content)
        print("----------------------------------")
