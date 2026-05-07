from build_vectorstore import build_indexing, build_reindexing
from query import query_index

index = build_indexing()

query = "Gradient Accumulation이 수학적으로 어떡해 가능한지 설명"

response = query_index(index, query)

print(response)