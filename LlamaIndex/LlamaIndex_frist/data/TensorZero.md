# TensorZero

ML 프로젝트와 서비스가 많이 개발되던 시기에 MLOps라는 분야가 생겼습니다. 기존의 DevOps라는 분야를 ML의 분야로 확장하는 분야로 생겼습니다. 항상 개발 시장에서 어떤 기술이 메인으로 자리를 잡으면 Ops라는 프로젝트는 계속해서 생기는 것 같습니다. 현재 개발 시장에서 주류로 평가는 받는 분야가 Agentic입니다. 좀 더 포괄적으로 얘기를 하자면 LLM이라고 할 수 있겠네요. 그렇다면 AgentOps나 LLMOps라는 분야도 있을 겁니다.

## LLMOps

LLMOps란 서비스의 전체 수명 주기 동안 AI 모델의 개발, 배포 및 관리를 가속화하는 전문적 사례 및 워크플로우를 의미압니다. LLMOps 플랫폼은 보다 효율적인 라이브러리 관리를 제공하므로 운영 비용을 절감하고 더 적은 기술 인력으로 작업을 완료할 수 있습니다. 기존의 MLOps와 같이 데이터 전처리, 학습, 파인튜닝, 모니터링 등의 기능을 구현합니다.

## Tensorzero란?

TensorZero는 위 LLMOps를 구축하는 하나의 서비스입니다. TensorZero는 LLM(거대 언어 모델)을 실제 서비스 환경에 배포할 때 발생하는 복잡성을 해결하기 위해 설계된 엔지니어링 중심의 오픈소스 Gateway입니다. 단순히 여러 모델을 연결하는 것을 넘어, 데이터 수집, 모델 최적화, 그리고 안정적인 운영을 위한 인프라 계층을 제공합니다.

## TensorZero의 메인 기능

### Gateway

TensorZero Gateway는 모든 LLM 애플리케이션을 위한 통합 인터페이스를 제공하는 고성능 모델 Gateway입니다. 또한 모든 LLM 애플리케이션을 위한 통합 인터페이스를 제공하는 고성능 모델 Gateway입니다. 모든 LLM을 위한 하나의 API. 이 Gateway는 모든 주요 LLM 제공업체를 위한 통합 인터페이스를 제공하여 원활한 플랫폼 간 통합 및 대체 기능을 지원합니다.

| 제공업체 | 대표 모델 | 특징 |
| --- | --- | --- |
| Anthropic | Claude 3 (Opus, Sonnet, Haiku) | 긴 컨텍스트, 안전성 중심 |
| Amazon Web Services Bedrock | Claude, Titan, Mistral 등 | 다양한 모델 통합 API |
| Amazon SageMaker | 커스텀 모델 | ML 파이프라인 구축 |
| Microsoft Azure | GPT-4, GPT-4o 등 | 엔터프라이즈 환경 강점 |
| Fireworks AI | Llama, Mistral 등 | 빠른 추론 속도 |
| Google Cloud Platform Vertex AI | Gemini 1.5 Pro 등 | Google 생태계 연동 |
| Google AI Studio | Gemini API | 빠른 프로토타이핑 |
| Groq | Llama 계열 | 초고속 inference |
| Hyperbolic | 다양한 OSS 모델 | 비용 효율성 |
| Mistral AI | Mistral, Mixtral | 오픈 가중치 모델 |
| OpenAI | GPT-4o, GPT-5 계열 | 범용 성능 최상위 |
| OpenRouter | 다양한 모델 라우팅 | 멀티 모델 선택 |
| Together AI | Llama, Mixtral 등 | 오픈소스 중심 |
| vLLM | 모든 Transformer 모델 | 고성능 서빙 엔진 |
| xAI | Grok | 실시간 정보 강점 |

[제공 가능한 모델 정보]

이 Gateway는 Rust 기반으로 구현되어 P99 기준 1ms 이하의 매우 낮은 지연 오버헤드를 가지며, 고부하 환경에서도 LiteLLM 대비 25~100배 이상 빠른 처리 성능을 제공합니다. 또한 입력과 출력에 스키마를 적용하는 구조화된 추론 방식을 통해 시스템 안정성을 높이고, 해당 데이터는 이후 프롬프트 최적화나 파인튜닝과 같은 고도화 작업에 활용될 수 있습니다.

여러 LLM 호출을 하나의 에피소드로 연결하는 다단계 워크플로를 지원하여 복잡한 Agent 시스템을 효과적으로 구성할 수 있으며, 추론 과정에서 발생하는 로그, 메트릭, 자연어 피드백 등을 자동으로 수집하는 관찰 가능성 기능을 통해 실시간 분석과 성능 개선이 가능합니다. 더불어 트래픽을 자동으로 분산시키는 A/B 테스트 기능을 제공하여 다양한 실험을 안정적으로 운영할 수 있고, 추론 실패 시 다른 모델이나 제공자로 자동 전환하는 fallback 기능을 통해 서비스 가용성을 유지합니다. 이 외에도 API Key 기반 접근 제어와 GitOps 방식의 설정 관리 기능을 지원하여 프롬프트, 모델, 파라미터, 실험 등을 체계적이고 확장 가능하게 운영할 수 있습니다.

### Observability

TensorZero Observability는 LLM 시스템의 모든 상호작용과 성능 지표를 투명하게 추적하는 실시간 모니터링 기반입니다. 또한 LLM 시스템의 모든 상호작용과 성능 지표를 투명하게 추적하는 실시간 모니터링 기반입니다. 데이터 기반 AI의 첫걸음. 이 기능은 추론 로그, 지연 시간, 비용, 피드백을 중앙 집중화하여 시스템의 상태를 한눈에 파악할 수 있게 해줍니다.

| **지표(Metric)** | **세부 항목** | **특징** |
| --- | --- | --- |
| **Latency** | TTFT(Time to First Token), Total Latency | 성능 병목 구간 파악 및 속도 최적화 |
| **Cost** | 토큰 사용량, 제공자별 청구 비용 | 리소스 추적 및 운영 예산 관리 |
| **Quality** | 유저 피드백, 모델 평가 지표 | 실제 서비스 만족도 및 품질 추적 |
| **Payload** | 프롬프트 입력, 모델 출력 및 스키마 로그 | 디버깅 및 파인튜닝용 데이터 수집 |
| **Integration** | ClickHouse, Datadog, Grafana | 기존 데이터 스택과 원활한 연동 |

이 Observability 기능은 단순히 API 호출 성공 여부를 넘어서, ClickHouse와 같은 고성능 데이터베이스에 모든 트레이스를 비동기적으로 기록하여 Gateway 자체의 짧은 지연 시간(P99 1ms 이하)에 영향을 주지 않고 데이터를 수집합니다. 수집된 데이터는 각 추론 단계별 토큰 사용량, 비용은 물론, 사용자 행동에서 발생하는 자연어 피드백이나 명시적인 평가 지표와 자동으로 매핑됩니다. 여러 LLM 호출을 하나의 에피소드로 연결하는 다단계 워크플로를 지원하여 복잡한 Agent 시스템의 맥락을 잃지 않고 효과적으로 디버깅할 수 있으며, 이 데이터들은 입력과 출력에 스키마를 적용하는 구조화된 방식을 통해 향후 프롬프트 최적화나 파인튜닝과 같은 고도화 작업에 즉각적으로 활용될 수 있습니다. 더불어 OpenTelemetry 표준을 지원하여 기존 인프라와 통합 운영이 가능하며, 실시간 분석을 통해 서비스 가용성을 유지하는 데 핵심적인 역할을 합니다.

### Optimization

TensorZero Optimization은 수집된 프로덕션 데이터를 바탕으로 LLM 애플리케이션의 성능을 극대화하고 비용을 절감하는 자동화 프로세스입니다. 또한 수집된 프로덕션 데이터를 바탕으로 LLM 애플리케이션의 성능을 극대화하고 비용을 절감하는 자동화 프로세스입니다. 피드백 루프를 통한 지속적 개선. 이 기능은 파인튜닝, 프롬프트 엔지니어링 및 동적 라우팅 파이프라인을 통해 가장 효율적인 추론 환경을 구축합니다.

| **최적화 기법** | **적용 대상** | **특징** |
| --- | --- | --- |
| **Fine-tuning** | 오픈소스 및 상용 모델 | 도메인 맞춤형 성능 향상 및 지연 시간 감소 |
| **Dynamic Routing** | 다중 모델 환경 | 작업 복잡도 및 트래픽에 따른 최적 모델 자동 할당 |
| **Prompt Engineering** | 입력 템플릿 및 시스템 프롬프트 | 구조화된 스키마를 통한 응답 정확도 개선 |
| **Distillation** | 거대 모델 → 소형 모델 | 고비용 모델의 능력을 저비용 모델로 전이하여 예산 절감 |
| **Data Extraction** | 프로덕션 로그 기반 데이터셋 | 성공적인 추론 사례를 수집하여 자동 훈련 데이터 구축 |

이 Optimization 과정은 기계적인 튜닝을 넘어, Observability를 통해 축적된 실제 서비스 데이터와 사용자 피드백을 직접적인 자양분으로 삼아 시스템 안정성과 응답 품질을 높입니다. 개발자는 데이터 추출 기능을 활용해 고품질의 성공 사례를 모아 파인튜닝 데이터셋을 손쉽게 생성할 수 있으며, 이를 바탕으로 비용이 많이 드는 대형 모델의 작업을 더 빠르고 저렴한 모델로 오프로드(Distillation)하여 성능 저하 없이 운영 비용을 획기적으로 낮출 수 있습니다. 입력과 출력에 엄격한 스키마를 적용하는 구조화된 추론 방식을 통해 최적화 과정 중에도 데이터의 일관성이 완벽하게 유지되며, 복잡한 프롬프트 수정 없이도 시스템이 더 나은 해결책을 도출하게 합니다. 나아가 추론 실패 시 다른 최적화된 모델로 자동 전환하는 fallback 기능을 결합하여, 성능 최적화가 서비스 가용성을 저해하지 않도록 체계적이고 확장 가능하게 운영할 수 있습니다.

### Evaluation

TensorZero Evaluation은 변경된 프롬프트나 새로운 파인튜닝 모델이 실제 서비스에 배포되기 전후의 품질을 체계적으로 검증하는 평가 프레임워크입니다. 또한 변경된 프롬프트나 새로운 파인튜닝 모델이 실제 서비스에 배포되기 전후의 품질을 체계적으로 검증하는 평가 프레임워크입니다. 데이터로 증명하는 모델 신뢰성. 이 기능은 정량적 메트릭과 LLM-as-a-Judge 방식을 결합하여 시스템의 응답 품질을 일관되게 측정합니다.

| **평가 방식** | **세부 지표 및 도구** | **특징** |
| --- | --- | --- |
| **Offline Evaluation** | 벤치마크, 골든 데이터셋 | 서비스 배포 전 안전성 및 정확도 사전 검증 |
| **Online Evaluation** | 실시간 로그, 유저 액션 맵핑 | 프로덕션 환경에서의 지속적인 성능 모니터링 |
| **LLM-as-a-Judge** | 정성적 품질, 톤앤매너, 유용성 평가 | 강력한 모델을 활용한 대규모 답변 자동 검수 |
| **Deterministic Metrics** | JSON 스키마 검증, 정규식 매칭 | 구조화된 출력의 형식 및 규칙 준수 여부 확인 |
| **Episode Tracking** | 다단계 워크플로 성공률 | Agent 시스템의 최종 목표 달성 여부 종합 평가 |

이 Evaluation 시스템은 개발자가 직관이나 감에 의존하지 않고, 데이터 기반으로 프롬프트와 모델 성능 차이를 명확히 비교할 수 있게 해줍니다. 과거의 에피소드 로그를 바탕으로 테스트 셋을 구성하고, 새로운 모델을 적용했을 때 출력값이 기존과 어떻게 달라졌는지, 지정된 스키마를 얼마나 정확히 준수하는지 정밀하게 추적합니다. 단순히 정형화된 텍스트 매칭뿐만 아니라, 뛰어난 추론 능력을 가진 모델을 평가자(Judge)로 활용하여 뉘앙스나 환각(Hallucination) 여부 등 정성적인 영역까지 자동화된 평가 파이프라인에 편입시킵니다. 관찰 가능성 기능에서 수집된 실제 사용자 피드백 메트릭과 연동되어 실시간 분석이 가능하며, 이를 통해 여러 단계로 구성된 복잡한 Agent 시스템 내에서도 병목 현상이나 품질 저하가 발생하는 지점을 정확히 짚어내어 플랫폼의 전반적인 신뢰성과 가용성을 높이는 데 기여합니다.

### Experimentation

TensorZero Experimentation은 다양한 모델, 프롬프트, 파라미터 조합을 프로덕션 환경에서 안전하게 테스트할 수 있는 강력한 실험 플랫폼입니다. 또한 다양한 모델, 프롬프트, 파라미터 조합을 프로덕션 환경에서 안전하게 테스트할 수 있는 강력한 실험 플랫폼입니다. 중단 없는 AI 시스템 혁신. 이 기능은 A/B 테스트와 카나리(Canary) 배포를 통해 위험을 최소화하면서 최적의 구성을 찾아냅니다.

| **실험 요소** | **세부 기능** | **특징** |
| --- | --- | --- |
| **A/B Testing** | 비율 기반 트래픽 분할 | 사용자 그룹별로 다른 모델이나 프롬프트 성능 비교 |
| **Variant Management** | GitOps 기반 버전 제어 | 구성 변경의 이력 추적, 리뷰 및 즉각적인 롤백 |
| **Canary Release** | 점진적 트래픽 롤아웃 | 소규모 트래픽으로 선제적 검증 후 전체 배포 |
| **Shadow Deployment** | 백그라운드 모델 추론 | 유저에게 응답하지 않고 새 모델의 성능 로그만 측정 |
| **Parameter Tuning** | Temperature, Top-p 등 동적 변경 | 환경 설정만으로 모델의 생성 세부 옵션 최적화 테스트 |

이 Experimentation 기능은 GitOps 방식의 설정 관리 기능을 철저히 지원하여, 코드를 직접 수정하지 않고도 설정 파일 변경만으로 새로운 모델, 프롬프트, 파라미터의 변형(Variant)을 체계적으로 운영하고 롤백할 수 있습니다. 트래픽을 자동으로 분산시키는 A/B 테스트를 통해 실제 사용자 피드백을 기반으로 어떤 구성이 더 나은 성과를 내는지 투명하게 증명하며, 백그라운드에서 추론을 실행하는 Shadow 테스트로 위험 부담 없이 대규모 실험을 진행할 수 있습니다. 실험 과정 중 특정 변형에서 에러가 발생하더라도, 사전에 정의된 fallback 기능을 통해 기존의 안정적인 모델로 즉시 전환되어 P99 기준 1ms 이하의 처리 성능과 서비스 가용성을 완벽하게 유지합니다. 여러 LLM 호출이 연결되는 복잡한 워크플로에서도 특정 노드만을 독립적으로 테스트할 수 있어, 변화하는 요구사항에 맞춰 모델을 지속적이고 안정적으로 교체하며 시스템을 확장해 나가는 데 필수적인 역할을 수행합니다.

## Tensorzero의 비전

TensorZero는 LLM 애플리케이션 최적화를 위한 데이터 및 학습 선순환 구조를 제공합니다. 이 피드백 루프는 실제 운영 환경에서의 지표와 사용자 피드백을 활용하여 더욱 스마트하고 빠르며 경제적인 모델과 에이전트를 생성합니다. 현재 TensorZero는 산업용 LLM 애플리케이션을 위한 오픈 소스 스택을 제공하며, 이는 LLM 게이트웨이, 관찰 가능성, 최적화, 평가 및 실험을 통합합니다. TensorZero의 비전은 LLM 엔지니어링의 많은 부분을 자동화하는 것이며, 오픈 소스 프로젝트를 통해 그 기반을 다지고 있습니다.

# Setting

### 1. 프로젝트 구조

```
project/
├── config/
│   └── tensorzero.toml
├── docker-compose.yml
├── app.py (or app.ts)
```

### 2. 핵심 설정 파일 (tensorzero.toml)

👉 최소 동작 세팅

```
# 함수 정의 (LLM task)
[functions.generate_haiku]
type = "chat"

# 사용할 모델 (variant)
[functions.generate_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini"
```

💡 핵심 구조

- function = 작업 단위 (예: 번역, 요약, 추출)
- variant = 모델 + 프롬프트 조합

➡️ 즉,

> "하나의 task에 여러 모델 실험 가능" 구조
> 

### 3. Docker 설정 (필수)

Quickstart 기준 핵심 구성:

```
services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      TENSORZERO_POSTGRES_URL: postgres://postgres:postgres@postgres:5432/tensorzero
    ports:
      -"3000:3000"

  ui:
    image: tensorzero/ui
    environment:
      TENSORZERO_GATEWAY_URL: http://gateway:3000
    ports:
      -"4000:4000"

  postgres:
    image: tensorzero/postgres:17
```

### 4. 실행

```
exportOPENAI_API_KEY=your_key
docker compose up
```

### 5. API 호출 (중요)

**기존 OpenAI 코드**

```
client=OpenAI()

response=client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role":"user","content":"hi"}]
)
```

**👉 TensorZero 적용 후**

```
fromopenaiimportOpenAI

client=OpenAI(
base_url="http://localhost:3000/openai/v1",
api_key="not-used"
)

response=client.chat.completions.create(
model="tensorzero::function_name::generate_haiku",
messages=[
        {"role":"user","content":"hi"}
    ],
)
```

✔ 핵심 변경점:

- `base_url` → Gateway로 변경
- `model` → `tensorzero::function_name::...`

➡️ 기존 코드 거의 그대로 유지 가능

### 6. UI

- 주소: `http://localhost:4000`
- 기능:
    - inference 로그 확인
    - metric / feedback 저장
    - fine-tuning 실행

➡️ 그냥 "LLM observability 대시보드"라고 보면 됨