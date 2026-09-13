# 반도체 제조 데이터 품질·불량 예측 검증 사례

## 프로젝트 요약

UCI SECOM의 590개 익명 공정 Feature로 생산 개체의 Pass(`-1`)와 Fail(`1`)을 분석했습니다. 1,567건 중 Fail은 104건(6.64%)뿐이고 결측·상수·중복·고상관 변수가 함께 존재했습니다.

목표는 높은 단일 수치를 만드는 데 그치지 않고, 데이터 품질부터 모델 평가, 운영 threshold, 시간 변화까지 **어떤 주장을 데이터가 실제로 지지하는지** 검증하는 것이었습니다.

## 수행한 방법

초기 Notebook 01~06에서 데이터 이해, 전처리, EDA, feature selection, 모델 비교와 설명 가능성 분석을 수행했습니다. 이후 검증상 취약점을 확인하고 다음 작업을 코드화했습니다.

1. 데이터 품질 규칙과 JSON·HTML 자동 리포트를 구현했습니다.
2. 결측률 제거, 상수 제거, median imputation, correlation pruning과 feature selection을 각 training fold 안에서 다시 학습했습니다.
3. 후보 4개를 동일한 5-fold × 5-repeat split에서 비교했습니다.
4. 각 outer training fold의 inner OOF 확률만으로 비용별 threshold를 고르고 별도 validation fold에서 평가했습니다.
5. Timestamp의 의미를 먼저 검토한 뒤 동일 timestamp를 나누지 않는 expanding-window 검증과 training 기준 PSI drift 분석을 추가했습니다.

Improvement 3~6에서는 기존 random 20% outer hold-out을 사용하지 않았습니다. 다만 historical Notebook은 이 hold-out을 이미 평가했으므로 프로젝트 전체에서 완전히 미개봉인 최종 test set이라고 주장하지 않습니다.

## 검증 결과

### Historical 결과와 fold-safe 결과

Historical Notebook의 CV PR-AUC는 0.274255, 당시 hold-out PR-AUC는 0.233475였습니다. Feature selection이 CV 바깥의 전체 outer-training set에서 먼저 수행됐으므로 0.274255를 개선된 fold-safe 성능으로 제시하지 않습니다.

전처리와 selection을 fold 내부로 이동한 단일 5-fold, 16조합 비교에서는 `all_features × random_forest_balanced`가 mean PR-AUC 0.219918을 기록했습니다.

### 반복 검증에서 확인한 작은 우위

| 후보 | Mean PR-AUC |
|---|---:|
| all-features + balanced RandomForest | 0.207031 |
| correlation-pruned + balanced RandomForest | 0.196414 |
| correlation-pruned + balanced HistGradientBoosting | 0.189125 |
| dummy prior | 0.066237 |

최선 후보와 correlation-pruned RF의 paired 차이는 +0.010617이고 25개 split에서 15승·10패였습니다. Training 표본이 겹치는 반복 fold를 독립 관측처럼 해석하지 않았고 RF의 우위를 확정적이라고 표현하지 않았습니다.

### 비용에 따라 달라지는 threshold

비용은 실제 제조 금액이 아닌 민감도 분석용 예시 단위입니다.

| FN:FP | 비용 선택 정책 | 기본 0.5/전부 정상 | 판단 |
|---|---:|---:|---|
| 1:1 | 85 | 83 | 기본 정책이 더 낮거나 같음 |
| 5:1 | 392 | 415 | 비용 선택 정책이 낮음 |
| 10:1 | 645 | 830 | 비용 선택 정책이 낮음 |

10:1 비용 선택 정책의 recall은 0.5181, precision은 0.1493이었습니다. 불량 누락 비용을 크게 둘수록 더 많은 확인 대상을 감수하는 정책이 선택됐습니다. Probability calibration은 하지 않았고 inner OOF와 outer 재학습 모델의 확률 척도가 달라질 수 있습니다.

### 시간순 평가가 드러낸 취약성

Timestamp 1,567건은 모두 파싱됐지만 timezone, 생산 순서 보장, lot·wafer 식별자가 없습니다. 따라서 “제한적으로 적합”하다고 판정하고 random outer-training 내부에서만 탐색적 시간순 평가를 수행했습니다.

세 forward split의 PR-AUC는 0.0857, 0.1848, 0.1008이었습니다. 고정 threshold 0.5에서는 모두 양성 예측이 0건이었습니다. Random repeated CV와 표본 구성·목적이 달라 동일 조건의 성능 하락률로 계산하지 않았습니다.

Training-only quantile bin으로 PSI를 계산했지만 PSI는 변화 신호일 뿐 공정 원인·설비 고장·인과관계의 증거가 아닙니다.

## 수행한 작업과 주장하지 않는 것

직접 구현·검증한 범위는 데이터 품질 리포트, fold-safe Pipeline, repeated CV, nested threshold 평가, timestamp 그룹 기반 forward split, PSI drift 요약, strict JSON·CSV 출력과 관련 회귀 테스트입니다.

반면 익명 Attribute만으로 온도·압력·장비 상태를 명명하거나 실제 수율 개선 효과를 계산하지 않았습니다. Lot·wafer·설비 정보가 없어 제조 단위 누수도 검증할 수 없습니다. 모델 결과와 SHAP·PSI를 실제 공정 원인으로 확대 해석하지 않았습니다.

## 프로젝트에서 내린 판단

- 높은 historical CV 수치보다 누수 가능성을 줄인 평가 범위를 우선했습니다.
- PR-AUC 1위 후보의 반복 우위가 작다는 사실을 그대로 유지했습니다.
- Threshold를 모델 고유의 정답이 아닌 FN/FP 비용 가정에 따른 운영 정책으로 다뤘습니다.
- 취약한 시간순 결과를 숨기지 않고 배포 전 추가 검증이 필요한 근거로 사용했습니다.

## 다음 단계

가장 우선할 작업은 새로운 모델 추가가 아니라 lot·wafer·설비·공정 단계 metadata 확보와 독립적인 최신 제조 기간 검증입니다. 이후 group-aware split, 외부 라인 평가, probability calibration 필요성 검토, training fold 내부 resampling 비교 순서로 확장할 수 있습니다.

구현과 재현 방법은 프로젝트 [README](../README.md)를 참고합니다.
