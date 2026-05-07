# 기존의 DL 튜닝의 문제점

딥러닝 모델을 학습할 때 가장 기본적인 튜닝 요소 중 하나는 batch size입니다. 하지만 batch size는 단순한 하이퍼파라미터가 아니라, 성능·속도·메모리를 동시에 결정하는 매우 중요한 요소입니다. 문제는 다음과 같습니다. batch size를 늘려야지 성능과 안정성이 올라가고 gradient variance이 감소합니다. 하지만 batch size가 늘어나면 GPU 메모리 사용량이 증가하고, OOM 발생 가능성이 높아집니다. batch size를 줄이면, 메모리는 적게 쓰지만 학습이 불안정해집니다. 즉, 좋은 성능을 위해 큰 batch를 쓰고 싶지만 GPU 메모리가 부족한 상황이 발생합니다.

## batch size

batch size는 한 번의 forward/backward에서 사용하는 데이터 샘플 수입니다. 예를 들어 batch size가 32라면 한 번에 32개 데이터로 gradient 계산한다는 말입니다. 당연하게 batch size가 크면 학습이 훨씬 안정적이겠지만, 메모리를 폭발적으로 사용하여서 OOM 오류를 발생시킵니다. 여기서 폭발적으로 증가시킨다는 것이 무슨 말이냐면, 단순히 batch size로 사용하는게 증가하는게 아니라 모델 파라미터, activation, optimizer state (특히 Adam 계열) 이것들이 모두 batch size에 비례해서 증가하게 됩니다.

# Gradient Accumulation

위의 DL의 성능을 늘려주고, OOM 에러는 해결하기 위해서 Gradient Accumulation이라는 기술을 사용해보기로 했습니다.

## Gradient Accumulation이란?

GPU 메모리 제한으로 인해 큰 배치를 한 번에 처리할 수 없을 때, 작은 미니 배치(mini-batch)에서 계산된 기울기(gradient)를 여러 번 누적한 뒤 한 번에 가중치를 업데이트하는 기술입니다. 즉, 기존에는 큰 배치 사이즈를 가지고 가중치를 업데이트 한다면, Gradient Accumulation은 미니 배치를 다시 잘라서 해당 기울기를 누적하고 가중치를 업데이트하는 방법입니다. 쉽게 말해서 일시불과 할부의 개념이라고 이해하시면 편합니다. 근데, 이러한 Gradient Accumulation이 수학적으로 가능한 걸까요?

## 이게 가능함?

Gradient Accumulation의 핵심은 바로 가중치를 구하는 과정이 선형적이라는 것 입니다. 수학에서 선형성이라는 말은 두가지를 만족한다는 의미입니다. 1. 덧셈 보존, 2. 스칼라 배 보존 입니다. 위 2가지를 간단하게 설명을 하면 f(x+y) = f(x) + f(y)가 덧셈 보존이고, f(a*x) = a*f(x)가 스칼라 배 보존입니다.

가중치 계산은 (L1 + L2 + L3 + L4)에 대하여 가중치를 계산하고 평균을 냅니다. 앞에서 말한 2가지 법칙 중에 2번 스칼라 배 보존을 사용하면 (∇L1+∇L2+∇L3+∇L4)의 평균을 구하는 식으로 표현이 가능합니다. 즉, 아래와 같은 수식이 성립을 하게 됩니다.

$$
\frac{1}{4}∇(L1+L2+L3+L4) = \frac{(∇L1+∇L2+∇L3+∇L4)}{4}
$$

즉, 각각의 배치에 대하여 Loss의 가중치를 구하고 평균을 내는 Gradient Accumlation이 가능하게 되는 겁니다. 하지만 이론적으로는 같으나 실제 환경에서는 아래의 것들 때문에 큰 배치와 같은 효과보다는 비슷한 효과를 낼 수 있다 정도로 이해하시면 됩니다.

### Gradient Accumulation 중 조심해야하는 것들

Gradient Accumulation은 이론적으로는 큰 batch와 동일한 효과를 낼 수 있지만, 실제 딥러닝 학습 환경에서는 완전히 동일하게 동작하지는 않습니다. 이는 모델의 구조나 학습 과정에서 batch에 의존하거나 상태(state)를 가지는 요소들 때문입니다. 대표적으로 아래 3가지 요소를 반드시 주의해야 합니다.

1. Batch Norm
    
    따라서 Batch Norm을 사용하는 경우에는 Gradient Accumulation 적용 시 몇 가지 추가적인 처리가 필요합니다.
    
    가장 간단한 방법은 BatchNorm을 사용하지 않는 것입니다. LayerNorm이나 GroupNorm과 같이 batch 통계에 의존하지 않는 정규화 기법을 사용하면, Gradient Accumulation 여부와 관계없이 안정적인 학습이 가능합니다.
    
    만약 BatchNorm을 반드시 사용해야 하는 경우에는, 통계의 불안정성을 줄이기 위한 방법을 고려해야 합니다. 예를 들어, 멀티 GPU 환경에서는 SyncBatchNorm을 사용하여 여러 GPU의 batch를 하나의 큰 batch처럼 통합하여 통계를 계산할 수 있습니다. 이를 통해 작은 batch로 인한 통계 왜곡 문제를 완화할 수 있습니다.
    
    또 다른 방법으로는 BatchNorm을 학습 도중에 고정(freeze)하는 방식이 있습니다. 일정 epoch 이후에는 BatchNorm을 eval 모드로 전환하여 running mean과 variance를 더 이상 업데이트하지 않도록 하면, 학습 안정성을 높일 수 있습니다.
    
    결론적으로, BatchNorm은 batch 크기에 직접적으로 의존하는 구조이기 때문에 Gradient Accumulation과 함께 사용할 경우 반드시 추가적인 설계가 필요하며, 그렇지 않으면 기대한 만큼의 성능을 얻기 어려울 수 있습니다.
    
2. optimizer
    
    Optimizer는 단순히 gradient만을 사용하는 것이 아니라, 내부적으로 다양한 상태(state)를 함께 유지합니다. 특히 Adam과 같은 optimizer는 1차 모멘트(momentum)와 2차 모멘트(variance), 그리고 step count 등을 기반으로 가중치를 업데이트합니다.
    
    문제는 Gradient Accumulation에서는 gradient가 여러 번 나누어 계산된 후 한 번에 optimizer에 전달된다는 점입니다.
    
    GA를 사용하면 여러 번의 backward를 통해 gradient가 누적된 뒤 한 번 update가 수행됩니다. 이 과정에서 optimizer는 “누적된 gradient”를 한 번에 받게 됩니다.
    
    반면 GA를 미사용 한다면, 큰 batch에서 계산된 gradient가 한 번에 들어오고, 그에 맞춰 optimizer 상태가 업데이트됩니다.
    
    즉, gradient의 값 자체는 유사할 수 있지만, optimizer 내부 상태가 업데이트되는 방식과 타이밍이 달라지게 됩니다. 그 결과, 동일한 데이터와 설정을 사용하더라도 학습 경로가 달라지고 최종 성능에도 영향을 줄 수 있습니다.
    
3. Dropout
    
    Dropout은 forward 과정에서 일부 뉴런을 랜덤하게 비활성화하여 과적합을 방지하는 기법입니다.
    
    문제는 Gradient Accumulation에서는 forward를 여러 번 수행한다는 점입니다.
    
    GA를 사용하면 각 미니배치마다 서로 다른 dropout mask가 생성되며, 여러 번의 forward 과정이 각각 다른 네트워크 구조에서 수행됩니다.
    
    반면 GA를 미사용 한다면, 큰 batch를 한 번에 처리하면서 하나의 dropout mask 기준으로 forward가 진행됩니다.
    
    즉, 동일한 데이터 16개를 사용하더라도 GA를 사용하는 경우에는 서로 다른 dropout 패턴을 거치게 되고, GA를 사용하지 않는 경우에는 동일한 패턴을 기준으로 처리됩니다.
    
    그 결과, 출력값과 gradient가 달라지게 되고, 학습 결과에도 차이가 발생할 수 있습니다.