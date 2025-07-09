from langchain_community.vectorstores import PathwayVectorClient

client = PathwayVectorClient(host="127.0.0.1", port=8666)
retriever = client.as_retriever()

results = retriever.invoke("your search query", k=1)
print(results)