import os
import warnings
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from openai import OpenAI

# ragas 0.4.x 호환: deprecated import 경고 숨김
warnings.filterwarnings(
    "ignore",
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
    category=DeprecationWarning,
)

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

# 1. .env 파일 로드
load_dotenv()

# OpenAI client / LLM / Embeddings
openai_client = OpenAI()
llm = llm_factory("gpt-4o-mini", client=openai_client)
embeddings = LangchainOpenAIEmbeddings(model="text-embedding-3-small")

# 2. 평가 데이터 준비 (정답셋 없음)
data_without_gt = {
    "question": ["RAG 시스템에서 구조적 환각을 통제하려면 어떻게 해야 하나요?"],
    "contexts": [["문서를 정확하게 검색했더라도 LLM이 온전히 반영한다는 보장이 없으므로, 모델이 제공된 맥락을 충실히 따르는지 검증해야 합니다. 이를 위해 Faithfulness 지표를 활용합니다."]],
    "answer": ["LLM이 검색된 맥락을 충실히 따르는지 검증하는 Faithfulness 지표를 통해 환각을 통제할 수 있습니다."]
}

dataset_no_gt = Dataset.from_dict(data_without_gt)

# 3. 평가 실행
result_no_gt = evaluate(
    dataset = dataset_no_gt,
    llm=llm,
    embeddings=embeddings,
    metrics=[
        faithfulness,
        answer_relevancy,
    ],
)

# 4. 결과 출력
print("=== 정답셋이 없는 경우의 평가 결과 ===")
print(result_no_gt)