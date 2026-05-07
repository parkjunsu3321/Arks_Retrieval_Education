# LlamaIndex

LLM(대규모 언어 모델)과 사용자의 개별 데이터를 연결하기 위한 강력한 데이터 프레임워크입니다. RAG(Retrieval-Augmented Generation) 시스템을 구축할 때 핵심적인 역할을 하며, 데이터를 수집하고 구조화하여 LLM이 활용할 수 있게 돕습니다.

## 기존의 프레임 워크와 다른점

![image.png](https://private-user-images.githubusercontent.com/127470862/588639499-6f9fdbe9-d64c-4724-9904-dede59908aa8.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzgxMjA4NjIsIm5iZiI6MTc3ODEyMDU2MiwicGF0aCI6Ii8xMjc0NzA4NjIvNTg4NjM5NDk5LTZmOWZkYmU5LWQ2NGMtNDcyNC05OTA0LWRlZGU1OTkwOGFhOC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTA3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUwN1QwMjIyNDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00Yzc2ZjVlODQyNTk2YmFkYTM4NGE2NjgwMzgxMWZkMzFhZWE4NzhlY2IwZTI5ZWY4MzM0OWY0OTMxMWMwNGQzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.00GVbi3_l40AKBj7B_BGQ5AW-bXBxiqWlTJwMGGQlpo)

[https://medium.com/@ranapratapdey/llm-orchestration-langchain-vs-llamaindex-vs-haystack-41622385eced]

위 3가지 프레임 워크가 대표적인 프레임 워크들 입니다. haystack, LangChain, LlamIndex순으로 개발이 되었습니다. 각각의 특징이 있습니다.

haystack은 검색 및 Q&A에 특화되어 있습니다. 비교적 다른 프레임 워크에 비해서 적용범위가 너무 좁으며, 19년도에 개발되었어서 비교적 오래된 프레임 워크입니다.

LangChain은 범용성이 높은 LLM 앱 프레임 워크입니다. 대부분의 LLM 프로젝트는 LangChain을 사용합니다. LLM 앱 개발 Lifecycle을 간소화한 프레임 워크입니다. 또한 다양한 라이브러리와 통합이 가능합니다.

마지막으로 LlamaIndex는 RAG 특화 프레임 워크입니다. RAG 커스텀이 쉽고 자유도가 높으며, 다양한 데이터 소스 통합이 가능합니다.

# LlamaIndex 구성 요소

![image.png](https://private-user-images.githubusercontent.com/127470862/588639558-16e3150d-8454-4da1-bc51-3a3c2fd6fc3c.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzgxMjA4NjIsIm5iZiI6MTc3ODEyMDU2MiwicGF0aCI6Ii8xMjc0NzA4NjIvNTg4NjM5NTU4LTE2ZTMxNTBkLTg0NTQtNGRhMS1iYzUxLTNhM2MyZmQ2ZmMzYy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTA3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUwN1QwMjIyNDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iMjczYWYwNzczNGJlMWM0MjBjNGM2ODRjZGRkNmMwZjdiMTQyZWU5NjA0MjI1YWRlZmM4ODU2MmFmZjRjMWVkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.Hm6otMdblXkpibrXdwNpZ2pOqrt2u8Fpu6EQb6ByjI4)

[https://www.geeksforgeeks.org/machine-learning/what-is-llamaindex/]

## Loading

데이터 로딩(Loading)은 외부의 다양한 데이터 소스(API, PDF, 웹사이트, 데이터베이스 등)에 흩어져 있는 데이터를 LLM이 사용할 수 있도록 가져와 하나의 일관된 형태로 변환하는 과정입니다. 이때 데이터 커넥터(Loader 또는 Reader)를 통해 원본 데이터를 수집하고, 본문과 함께 출처·생성일 같은 메타데이터를 포함한 ‘문서(Document)’ 형태로 변환합니다. LlamaIndex는 다양한 형식의 데이터를 자동으로 처리할 수 있는 리더를 제공하며, 추가적으로 LlamaHub를 통해 수백 개의 커넥터를 활용해 거의 모든 데이터 소스를 연결할 수 있습니다. 이렇게 로딩된 데이터는 이후 인덱싱과 검색을 위한 지식 베이스의 기반이 됩니다.

## Indexing

인덱싱(Indexing)은 수집된 데이터를 LLM이 빠르고 정확하게 검색할 수 있도록 구조화하는 과정으로, 데이터를 청크 단위로 나누고 이를 벡터 임베딩으로 변환해 의미 기반 검색이 가능하도록 만든다. LlamaIndex에서는 `Document`를 `Node`로 분할한 뒤 각 Node를 벡터로 변환하여 `VectorStoreIndex`에 저장하며, 이때 메타데이터도 함께 관리되어 필터링과 다양한 검색 전략을 지원한다. 사용자의 쿼리 역시 동일하게 벡터로 변환된 후 유사도 계산을 통해 가장 관련성이 높은 데이터를 찾는 방식으로 동작하며, 이러한 인덱스는 메모리나 벡터 데이터베이스에 저장되어 효율적인 재사용이 가능하다.

## Storing

데이터 저장(Storage)은 인덱싱된 데이터를 재사용할 수 있도록 보관하는 과정으로, 다시 인덱싱하지 않고도 빠르게 검색과 질의를 가능하게 만드는 단계이다. LlamaIndex는 문서(Document), 인덱스 메타데이터, 벡터 임베딩 등을 각각 Document Store, Index Store, Vector Store 등에 나누어 저장하며, 필요에 따라 지식 그래프나 채팅 데이터도 별도로 관리할 수 있다. 이러한 데이터는 메모리뿐 아니라 로컬 디스크나 Amazon S3, Cloudflare R2 같은 외부 스토리지에 저장할 수 있어, 한 번 구축한 인덱스를 지속적으로 활용할 수 있게 해준다.

## **Quering**

쿼리(Querying)는 인덱싱된 데이터를 기반으로 사용자의 자연어 질문에 대해 최적의 답을 생성하는 과정으로, 검색(Retrieval), 후처리(Post-processing), 응답 합성(Response Synthesis)의 흐름으로 이루어진다. 먼저 검색 단계에서는 사용자의 쿼리를 벡터 임베딩으로 변환한 뒤 인덱스에서 의미적으로 가장 유사한 데이터(Node)를 top-k 방식으로 가져오고, 이후 후처리 단계에서는 재순위화, 필터링, 변환 등을 통해 불필요한 정보를 제거하고 더 적절한 컨텍스트로 정제한다. 마지막으로 응답 합성 단계에서는 이렇게 선별된 데이터와 사용자의 질문을 결합하여 LLM에 전달하고, 이를 바탕으로 최종 응답을 생성한다. LlamaIndex의 쿼리 엔진은 이러한 전체 흐름을 추상화한 핵심 컴포넌트로, 하나 이상의 인덱스와 검색기를 조합하여 동작하며 필요에 따라 하위 쿼리, 다단계 추론, 하이브리드 검색 등의 전략을 적용할 수 있다. 또한 단순 질의응답을 넘어 채팅 엔진, 에이전트, 라우터, 노드 후처리기 등 다양한 모듈과 결합하여 더 복잡하고 지능적인 질의 처리 파이프라인을 구성할 수 있다.

# 예시 코드

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```