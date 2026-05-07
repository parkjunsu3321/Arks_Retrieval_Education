from build_vectorstore import build_indexing, build_reindexing
# 1. vectorstore/index를 만들었다고 가정
index = build_indexing()

def query_index(index, query):
    query_engine = index.as_query_engine(
        similarity_top_k=3
    )

    response = query_engine.query(query)

    return response