import os
import warnings
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

# ragas 0.4.x에서만 필요한 deprecation 경고 숨김
warnings.filterwarnings(
    "ignore",
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
    category=DeprecationWarning,
)

from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
)

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI, AsyncOpenAI

load_dotenv()

# OpenAI client
openai_client = OpenAI()
async_openai_client = AsyncOpenAI()

# LLM
llm = llm_factory(
    "gpt-4o-mini",
    client=openai_client
)

# embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    client=async_openai_client,
)

# 데이터
data_with_gt = {
    "question": ["RAG 평가는 왜 어려운가요?"],
    "contexts": [[
        "RAG 평가는 정답셋 구축 비용의 한계와 검색/생성을 모두 다뤄야 하는 다차원적 복합 평가라는 점 때문에 어렵습니다."
    ]],
    "answer": [
        "RAG 평가는 주로 정답셋의 부재와 검색 및 생성을 동시에 평가해야 하는 복잡성 때문에 어렵습니다."
    ],
    "ground_truth": [
        "RAG 평가는 정답 데이터 구축의 현실적인 어려움과, 검색 성능과 생성 성능을 분리해서 복합적으로 평가해야 하는 특징 때문에 본질적으로 어렵습니다."
    ]
}

dataset = Dataset.from_dict(data_with_gt)

# 평가
result = evaluate(
    dataset=dataset,
    llm=llm,
    embeddings=embeddings,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_correctness,
    ],
)

print(result)