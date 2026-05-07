# LlamaIndex_frist

`data` 폴더의 문서를 벡터 인덱스로 저장하고, 저장된 인덱스를 기반으로 질의응답을 수행하는 간단한 LlamaIndex 실습 프로젝트입니다.

## 프로젝트 구조

- `main.py`  
  인덱스를 준비하고 질의를 실행하는 시작점 스크립트입니다.
- `build_vectorstore.py`  
  인덱스 생성/재사용/재생성 로직을 담당합니다.
- `query.py`  
  Query Engine 생성과 질의 실행 함수를 제공합니다.
- `data/`  
  인덱싱 대상 문서(마크다운) 폴더입니다.
- `store/`  
  인덱스 저장 결과(`docstore.json`, `index_store.json` 등)가 생성되는 폴더입니다.

## 주요 함수

### `build_indexing()`

- `store`에 저장된 인덱스가 있으면 새로 빌드하지 않고 로드합니다.
- 저장된 인덱스가 없으면 `data` 문서를 읽어 새 인덱스를 만들고 `store`에 저장합니다.

### `build_reindexing()`

- 기존 `store`를 삭제한 뒤 인덱스를 강제로 다시 생성합니다.
- 문서가 변경되었을 때 최신 상태로 재인덱싱할 때 사용합니다.

### `query_index(index, query)`

- 전달받은 인덱스로 Query Engine을 생성합니다.
- `similarity_top_k=3` 설정으로 유사한 문서를 검색한 뒤 응답을 반환합니다.

## 실행 방법

프로젝트 루트에서 아래 명령으로 실행합니다.

```powershell
python .\LlamaIndex_frist\main.py
```

기본 실행 흐름:
1. `build_indexing()`으로 인덱스 로드 또는 생성
2. 질문 문자열 준비
3. `query_index()` 호출
4. 응답 출력

## 재인덱싱 방법

문서를 수정한 뒤 기존 인덱스를 무시하고 새로 만들고 싶다면 `build_reindexing()`을 호출하도록 실행 코드를 바꾸면 됩니다.

예: `main.py`에서
- `index = build_indexing()` 대신
- `index = build_reindexing()` 사용