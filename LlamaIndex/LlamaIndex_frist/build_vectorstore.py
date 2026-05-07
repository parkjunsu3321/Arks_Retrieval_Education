import os
import shutil
os.environ["OPENAI_API_KEY"] = "sk-proj-1234567890"

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
store_dir = os.path.join(base_dir, "store")

def build_indexing():
    has_store_data = os.path.isdir(store_dir) and any(os.scandir(store_dir))

    if has_store_data:
        print("store 데이터가 이미 존재하여 빌드를 건너뛰고 로드합니다.")
        storage_context = StorageContext.from_defaults(persist_dir=store_dir)
        return load_index_from_storage(storage_context)

    print("store 데이터가 없어 새로 빌드합니다.")
    docs = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(docs)
    os.makedirs(store_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=store_dir)
    return index

def build_reindexing():
    if os.path.isdir(store_dir):
        shutil.rmtree(store_dir)

    print("기존 store를 삭제하고 인덱스를 다시 빌드합니다.")
    docs = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(docs)
    os.makedirs(store_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=store_dir)
    return index