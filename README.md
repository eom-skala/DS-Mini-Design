# ESS 배터리 수명 예측 
배터리 셀의 초기 100사이클 방전 용량 곡선으로 최종 Cycle Life를 예측한다. Batch 1에서 모델을 학습·선택하고, Batch 2에서 배치 간 일반화 성능을 최종 평가한다.

---
## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)
- 태스크 : Regression (Cycle Life 예측)
- Target : `cycle_life` — 원자료에 기록된 EOL 도달까지의 총 사이클 수
- 입력 피처 : `dq_min`, `log_dq_var`
- 검증 데이터 : Batch 1 내 충전 프로토콜 그룹 Hold-out
- 평가 지표 : MAPE (%), 원논문 참고값 9.1%

---
## INPUT

### 예측 목적과 분석 단위

초기 100사이클 측정이 끝난 셀의 **총 Cycle Life**를 예측해, 수명이 짧을 가능성이 있는 셀의 추가 검사와 점검 우선순위 선정에 활용한다. 학습 테이블의 한 행은 한 셀이며, 예측 시점은 사이클 100 종료 시점이다. 총 수명과 현재 잔여수명은 구분한다. 현장 적용 가능성은 별도 검증이 필요하다.

### 데이터 수집·정리와 누락 확인

- 제공된 MIT-Stanford 배터리 실험 `.mat` 데이터를 사용한다. 이 프로젝트에서 현장 BESS 데이터를 새로 수집한 것은 아니다. 충전 프로토콜별 셀 수명 차이와 초기 방전 곡선 변화가 수명을 설명할 수 있다는 가설에서 출발한다.
- Batch 1은 학습·모델 선택, Batch 2는 최종 평가, Batch 3는 설계서의 EDA 비교용이다. Batch 1은 46셀, Batch 2는 47셀 중 수명이 확인된 39셀, Batch 3는 44셀 기준으로 EDA를 해석한다. 이미지별 분석 가능 셀 수는 결측·그룹 조건에 따라 달라질 수 있다.
- `summary['cycle']`의 실제 번호로 10·100사이클을 찾는다. 단순 배열 인덱스를 사이클 번호로 간주하지 않는다. `Qdlin`과 `Vdlin` 길이, 유효값, 전압 중복, summary와 cycles의 대응을 확인한다.
- 타깃 누락·무효, 100사이클 이전 EOL, 필요한 사이클 번호 누락은 제외 사유를 기록한다. 곡선 오류는 피처 결측으로 기록하고 학습 부분의 중앙값으로 대체한다. 충전 프로토콜을 식별할 수 없으면 그룹 분할이 불가능하므로 실행을 중단한다.
- 수명 누락을 짧은 수명이나 0으로 채우지 않는다. 미도달 EOL에 따른 관측 중단인지 기록을 확인해야 하며, 그렇다면 일반 회귀에서 제외한 표본에 선택 편향이 있을 수 있다.

### X·Y 정의와 변수 선정 이유

| 구분 | 변수·원자료 | 의미와 선정 이유 | 모델에서의 역할 |
| --- | --- | --- | --- |
| Y | `cycle_life` | 원자료에 기록된 EOL까지의 총 사이클 수. 셀의 사용 가능 수명을 비교하려는 목적과 연결된다. | 양수 연속값을 예측하는 회귀 타깃 |
| X 원자료 | 사이클 10·100의 `Qdlin`, `Vdlin` | 동일 전압에서 초기 용량 변화량을 계산할 수 있고 사이클 100 시점에 확보 가능하다. | `ΔQ(V) = Q100(V) − Q10(V)` 생성 |
| X 피처 | `dq_min` | ΔQ 곡선의 가장 깊은 음의 변화. EDA에서 관찰한 감소 크기를 한 값으로 요약한다. | 최종 입력, Ah |
| X 피처 | `log_dq_var` | 한 셀의 ΔQ 곡선 내부에서 전압별 변화량이 얼마나 퍼지는지 요약한다. 작은 분산의 크기 차이를 로그로 표현한다. | 최종 입력, `log10(var(ΔQ) + 1e-12)` |
| 분할·감사 | 셀 ID, Batch, 충전 프로토콜, 바코드 | 셀·프로토콜 중복과 배치 간 차이를 확인한다. 셀 ID 자체는 수명 설명 변수가 아니다. | 입력에서 제외 |
| 추가 실험 후보 | 초기 충전시간·온도, 단계별 C-rate·전환 SOC | EDA에서 연관성이나 조건 차이가 관찰됐으나 Batch별 방향·측정 정의를 검증해야 한다. | 현재 모델에는 사용하지 않음 |

EOL의 용량 기준과 측정 조건은 원자료·실험 기록으로 확인한다. EDA 그림의 0.85·0.86·0.88 Ah 선을 일괄 타깃 정의로 사용하거나 곡선 끝을 EOL로 재정의하지 않는다. 전체 열화 속도, 후기 Knee, 최종 관측 길이, 장·단수명 그룹은 예측 시점 이후 정보 또는 타깃 파생 정보이므로 X에서 제외한다.

---
## 파일 구조 (sample) 
```text
├── images/                         
│   ├── Batch1_*.png
│   ├── Batch2_*.png
│   └── Batch3_*.png
├── ess-battery-project/
│   ├── DS-MINI-Design-울산_3반-엄진용.ipynb
│   ├── DS-MINI-Design-울산_3반-엄진용.before-modeling.ipynb
│   ├── run_project.py              # 단일 파일 실행: EDA → Train/CV → Valid → Test
│   └── model_outputs/
│       ├── batch1_split_manifest.csv
│       ├── batch1_nested_cv.csv
│       ├── batch1_candidate_performance.csv
│       ├── train_feature_audit.csv
│       ├── selection_lock.json
│       ├── selected_model_performance.csv
│       ├── batch2_predictions.csv
│       ├── batch2_exclusions.csv
│       ├── batch2_final_test.png
│       ├── run_environment.json     # Python 실행 시 생성
│       └── eda/                    # Python 실행 시 생성되는 Batch 1 EDA 표·그래프
├── .env                            # 로컬 데이터 경로 설정 (Git 제외)
├── .env.example                    # 공유용 설정 예시
├── requirements.txt
└── README.md
```


## 환경 설정 (sample) 
Python 3.11 또는 3.12를 사용한다. `requirements.txt`는 두 버전에서 설치할 수 있는 패키지 버전을 고정한다. 기존 성능표는 이전 검증 환경의 결과이므로 패키지 버전 변경 시 실행 결과에 차이가 날 수 있다.

프로젝트 루트의 `.env` 파일에서 데이터 폴더 경로를 지정한다. Python 파일이나 터미널 인자에서 데이터 경로를 수정할 필요가 없다.

```dotenv
# .env: 상대 경로는 프로젝트 루트 기준
DATA_DIR="../Data"
```

- 현재 `.env`는 프로젝트와 나란히 있는 `Data` 폴더를 가리킨다. 다른 환경에서는 `DATA_DIR`을 실제 폴더 경로로 수정한다.
- 절대 경로도 사용할 수 있다. 공백이 포함된 경로는 위처럼 따옴표로 감싼다.
- 새로 복제한 저장소에는 `.env.example`을 `.env`로 복사한 뒤 경로를 설정한다. `.env`는 기존 `.gitignore`에 따라 Git에서 제외된다.
- 실행 위치와 관계없이 Python 파일이 프로젝트 루트의 `.env`를 읽는다. 원본 데이터는 저장소에 포함하지 않는다.

```bash
git clone https://github.com/eom-skala/DS-Mini_Design DS-Mini-Design
cd DS-Mini-Design
python3.11 -m venv .venv
source .venv/bin/activate
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
# .env의 DATA_DIR을 실제 데이터 폴더 경로로 수정
./.venv/bin/python ess-battery-project/run_project.py
```


- Batch 1 파일: `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- Batch 2 파일: `2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- 기본 결과 경로는 `ess-battery-project/model_outputs/`이다. 다른 경로는 `--output-dir`로 지정한다.
- Python 파일은 노트북의 로딩·요약 통계·ΔQ 특징 분석과 모델 설계를 단일 실행으로 정리했다. 전체 raw 시계열 대신 필요한 필드를 읽고, EDA는 모델에 반영하지 않는 보고용으로 생성한다.
- CV·Hold-out·최종 Test는 노트북과 같은 두 피처, seed, 후보 모델, 탐색 범위를 사용한다. 모델 선택을 끝낸 뒤 Batch 2를 한 번만 평가하며, 한 번의 실행 안에서 Test를 튜닝에 사용하지 않는다.
- README의 Batch 1·2·3 이미지는 설계서의 기존 EDA 결과다. Python 파일의 EDA 출력은 제공된 노트북 범위인 Batch 1이며 Batch 3를 학습에 사용하지 않는다.
- 노트북을 별도로 실행하려면 기존 EDA에 사용하는 `mat73`와 Jupyter를 추가 설치하고 `DATA_DIR`을 설정한다. 단독 Python 실행에는 이 두 패키지가 필요하지 않다.

```bash
# 모델 평가만 실행하고 Batch 1 EDA 출력은 생략
./.venv/bin/python ess-battery-project/run_project.py --skip-eda

# Batch 2를 열지 않고 Batch 1 실행만 확인
./.venv/bin/python ess-battery-project/run_project.py --train-only

# 별도 결과 폴더로 출력
./.venv/bin/python ess-battery-project/run_project.py --output-dir "./results/run1"
```

---
## EDA


### 1. Cycle Life 분포와 이상치

- **분석 목적:** 학습 범위가 평가 셀의 수명을 포함하는지, 짧은 수명이 드문 이상치인지 또는 한 배치의 주요 집단인지 확인한다.
- **관찰 결과:** Batch 1 평균은 약 845사이클이며 <500사이클 셀은 없고 >1,000사이클은 10/46셀(21.74%)이다. Batch 2는 수명 누락 8셀을 제외하면 평균 약 566사이클이며 단수명은 28/39셀(71.79%), 장수명은 3/39셀(7.69%)이다. Batch 3 평균은 약 1,060사이클이며 단수명은 없고 장수명은 23/44셀(52.27%)이다.
- **처리 결정과 전후 차이:** 누락 수명을 제거한 뒤 박스 플롯과 비율을 계산한다. Batch 2의 비율 분모는 전체 47셀이 아니라 수명이 확인된 39셀이다. IQR 바깥의 셀은 측정 오류가 확인되지 않으면 유지한다. 이상치 삭제나 타깃 로그 변환으로 성능이 좋아졌다는 결과는 없다.
- **모델링 시사점:** Batch 1에 없는 단수명 영역을 Batch 2에서 예측해야 하므로 수명 과대예측 위험이 있다. MAPE는 짧은 수명 셀의 같은 절대 오차를 더 크게 반영하므로 전체 MAPE와 셀별 APE를 함께 해석한다. 후속 개선에는 단수명 학습 셀과 새로운 평가 배치를 확보한다. 타깃 분포만 보고 특정 확률분포를 가정하지 않으며, 현재 구현은 원래 사이클 단위의 회귀를 사용한다.

<details>
<summary>수명 분포·비율·이상치 이미지</summary>

#### Batch 1

**Histogram**

<img src="images/Batch1_Histogram.png" alt="Batch 1 Histogram" width="100%">

**장·단수명 비율**

<img src="images/Batch1_BatteryProportion.png" alt="Batch 1 장·단수명 비율" width="100%">

**Box Plot**

<img src="images/Batch1_BoxPlot.png" alt="Batch 1 Box Plot" width="100%">

#### Batch 2

**Histogram**

<img src="images/Batch2_Histogram.png" alt="Batch 2 Histogram" width="100%">

**장·단수명 비율**

<img src="images/Batch2_BatteryProportion.png" alt="Batch 2 장·단수명 비율" width="100%">

**Box Plot**

<img src="images/Batch2_BoxPlot.png" alt="Batch 2 Box Plot" width="100%">

#### Batch 3

**Histogram**

<img src="images/Batch3_Histogram.png" alt="Batch 3 Histogram" width="100%">

**장·단수명 비율**

<img src="images/Batch3_BatteryProportion.png" alt="Batch 3 장·단수명 비율" width="100%">

**Box Plot**

<img src="images/Batch3_BoxPlot.png" alt="Batch 3 Box Plot" width="100%">


</details>

### 2. 열화 곡선과 Knee

- **분석 목적:** 열화가 일정한 기울기로 진행되는지, 셀마다 가속 시작 시점이 달라지는지 확인한다.
- **관찰 결과:** 많은 셀은 초기 안정화·완만한 감소 이후 후기 감소 속도가 커진다. 일부는 관측 종료까지 뚜렷한 가속이 없다. Knee 후보는 Batch 2에서 약 250 ~ 400, 600 ~ 750, 800 ~ 1,000사이클, Batch 3에서 약 300 ~ 600, 600 ~ 1,000, 1,200 ~ 1,700사이클에 보인다. 이는 곡선에서 읽은 대략적인 범위로, 셀별 Knee 검출 결과가 아니다.
- **처리 결정과 전후 차이:** 곡선의 순간 급락과 지속적인 기울기 변화를 구분한다. 용량 필터로 잘린 곡선 끝을 EOL이나 Knee로 확정하지 않는다. 필터 전후의 Knee 정확도는 정량 검증하지 않았다.
- **모델링 시사점:** 후기 열화는 단순하지 않지만 이것만으로 초기 피처와 수명 사이의 비선형성을 확정할 수는 없다. 선형 모델과 비선형 후보를 비교하되, 전체 열화 속도와 후기 Knee는 미래 정보이므로 초기 예측 입력에 넣지 않는다.

<details>
<summary>사이클별 방전 용량 곡선</summary>

#### Batch 1

**Qd 곡선**

<img src="images/Batch1_QdCurve.png" alt="Batch 1 Qd 곡선" width="100%">

#### Batch 2

**Qd 곡선**

<img src="images/Batch2_QdCurve.png" alt="Batch 2 Qd 곡선" width="100%">

#### Batch 3

**Qd 곡선**

<img src="images/Batch3_QdCurve.png" alt="Batch 3 Qd 곡선" width="100%">


</details>

### 3. 초기 ΔQ(V) 형태와 피처 추출

- **분석 목적:** 사이클 100 시점에 이미 확보된 변화량으로 장·단수명 셀을 구분할 가능성이 있는지 확인한다.
- **관찰 결과:** Batch 2의 단수명 28셀은 장수명 3셀보다 깊은 음의 골과 넓은 셀 간 편차를 보인다. 그룹 중앙값 곡선의 최저점은 약 −0.055 Ah와 −0.02 Ah이다. Batch 1·3에는 <500사이클 셀이 없어 동일 기준의 장·단수명 비교가 불가능하다.
- **처리 결정과 전후 차이:** 원래의 긴 전압별 곡선을 `dq_min`과 `log_dq_var` 두 값으로 요약해 셀별 고정 길이 입력을 만든다. 로그는 ΔQ의 음수 값에 적용하지 않고 양수인 분산에 적용한다. 그림의 그룹 간 IQR은 셀 사이 변동이고, `log_dq_var`는 한 셀 안의 전압별 변동이므로 둘을 구분한다.
- **모델링 시사점:** 감소 크기와 곡선 형태를 적은 변수로 표현할 수 있어 소표본 회귀에 적합한 후보 피처다. 다만 Batch 2 장수명 표본이 3셀뿐이므로 구분 성능이나 인과관계가 입증된 것은 아니다. 두 피처의 유효성은 Batch 1 그룹 CV에서 평가하고, 각 학습 부분에서 공통 전압 격자를 정한다.

<details>
<summary>ΔQ 곡선과 셀별 통계</summary>

#### Batch 1

**ΔQ(V) 개별 곡선**

<img src="images/Batch1_IndQdCurves.png" alt="Batch 1 ΔQ(V) 개별 곡선" width="100%">

**ΔQ 셀별 통계**

<img src="images/Batch1_QdStatistics.png" alt="Batch 1 ΔQ 셀별 통계" width="100%">

#### Batch 2

**ΔQ(V) 개별 곡선**

<img src="images/Batch2_IndQdCurves.png" alt="Batch 2 ΔQ(V) 개별 곡선" width="100%">

**ΔQ 셀별 통계**

<img src="images/Batch2_QdStatistics.png" alt="Batch 2 ΔQ 셀별 통계" width="100%">

#### Batch 3

**ΔQ(V) 개별 곡선**

<img src="images/Batch3_IndQdCurves.png" alt="Batch 3 ΔQ(V) 개별 곡선" width="100%">

**ΔQ 셀별 통계**

<img src="images/Batch3_QdStatistics.png" alt="Batch 3 ΔQ 셀별 통계" width="100%">


</details>

### 4. 충전 프로토콜·전류 패턴과 수명

- **분석 목적:** 최대 전류 하나로 수명을 설명할 수 있는지, 프로토콜 차이가 분할과 일반화에 어떤 영향을 주는지 확인한다.
- **관찰 결과:** Batch 1의 `8C(15%)-3.6C`와 `8C(35%)-3.6C`는 첫 단계 C-rate가 같지만 평균 수명은 약 1,008.5와 607.5사이클로 다르다. Batch 2는 같은 프로토콜 표기에서도 `newstructure` 유무에 따라 차이가 있고, Batch 3에서는 첫 단계 C-rate가 가장 낮은 프로토콜이 최장수명은 아니다.
- **처리 결정과 전후 차이:** 셀을 개별 무작위 분할하는 대신 동일 충전 프로토콜을 한 그룹으로 묶어 Hold-out과 CV를 구성한다. 이렇게 하면 같은 프로토콜 셀이 학습·검증에 함께 들어가는 것을 방지한다. 개별 셀 분할 대비 성능 개선은 측정하지 않았다.
- **모델링 시사점:** 최대 전류만으로 고속 충전의 인과 효과를 단정하지 않는다. 전환 SOC·후속 전류·실험 구조가 함께 영향을 줄 수 있다. 프로토콜은 현재 분할용으로 사용하고 추가 입력은 별도 실험에서 검증한다. 전체 열화 속도와 전류의 상관은 설명용이며 예측 입력 선정에는 미래 정보 제약을 적용한다.

<details>
<summary>충전 프로토콜과 전류 패턴 분석</summary>

#### Batch 1

**프로토콜별 평균 수명 표**

<img src="images/Batch1_ChargeProtocol.png" alt="Batch 1 프로토콜별 평균 수명 표" width="100%">

**프로토콜별 평균 수명 그래프**

<img src="images/Batch1_MeanCycleLife.png" alt="Batch 1 프로토콜별 평균 수명 그래프" width="100%">

**전류 패턴과 전체 열화 속도**

<img src="images/Batch1_ChargePatternCorrelation.png" alt="Batch 1 전류 패턴과 전체 열화 속도" width="100%">

#### Batch 2

**프로토콜별 평균 수명 표**

<img src="images/Batch2_ChargeProtocol.png" alt="Batch 2 프로토콜별 평균 수명 표" width="100%">

**프로토콜별 평균 수명 그래프**

<img src="images/Batch2_MeanCycleLife.png" alt="Batch 2 프로토콜별 평균 수명 그래프" width="100%">

**전류 패턴과 전체 열화 속도**

<img src="images/Batch2_ChargePatternCorrelation.png" alt="Batch 2 전류 패턴과 전체 열화 속도" width="100%">

#### Batch 3

**프로토콜별 평균 수명 표**

<img src="images/Batch3_ChargeProtocol.png" alt="Batch 3 프로토콜별 평균 수명 표" width="100%">

**프로토콜별 평균 수명 그래프**

<img src="images/Batch3_MeanCycleLife.png" alt="Batch 3 프로토콜별 평균 수명 그래프" width="100%">

**전류 패턴과 전체 열화 속도**

<img src="images/Batch3_ChargePatternCorrelation.png" alt="Batch 3 전류 패턴과 전체 열화 속도" width="100%">


</details>

### 5. 초기 특징과 수명의 상관관계·다중공선성

- **분석 목적:** 유망한 추가 변수와 중복 정보를 찾고, 특정 배치의 상관을 다른 배치에 그대로 적용해도 되는지 확인한다.
- **관찰 결과:** 제시된 초기 요약 변수 중 평균 충전시간은 Batch 1·3에서 수명과 약 +0.58, +0.64로 가장 큰 절대 상관을 보인다. Batch 2에서는 평균 온도가 약 +0.42로 가장 크지만 Batch 1에서는 약 −0.48이다. 평균 온도와 최대 온도는 Batch 1·3에서 약 0.96~0.97로 매우 높다.
- **처리 결정과 전후 차이:** 평균·최대 온도를 동시에 추가하는 결정을 보류하고, 현재 입력은 두 ΔQ 피처로 제한한다. 기존 히트맵에는 `dq_min`, `log_dq_var`가 없으므로 이 두 피처의 공선성이 낮거나 높다는 결론을 내리지 않는다. 피처 제거 전후 성능이나 VIF 결과는 제시되지 않았다.
- **모델링 시사점:** 상관 방향의 배치 차이는 조건 의존성의 신호다. 추가 변수는 Train 내부에서만 선정하고, 두 ΔQ 피처의 상관·계수 안정성도 학습 데이터에서 확인해야 한다. Ridge는 공선성에 따른 계수 변동을 완화하는 후보이며, 정규화가 배치 간 분포 차이까지 해결하는 것은 아니다.

<details>
<summary>초기 100사이클 상관관계 Heatmap</summary>

#### Batch 1

**상관관계**

<img src="images/Batch1_CorrelationHeatmap.png" alt="Batch 1 상관관계" width="100%">

#### Batch 2

**상관관계**

<img src="images/Batch2_CorrelationHeatmap.png" alt="Batch 2 상관관계" width="100%">

#### Batch 3

**상관관계**

<img src="images/Batch3_CorrelationHeatmap.png" alt="Batch 3 상관관계" width="100%">


</details>

---
## Modeling

### 데이터 특성에 따른 문제 정의와 처리 전략

예측 시점은 사이클 100 종료 시점, Y는 총 Cycle Life, X는 `dq_min`·`log_dq_var`다. 최종 모델은 이 두 변수만 사용한다. EDA에서 보인 소표본, 프로토콜별 조건 차이, 배치 간 수명 분포 차이를 각각 모델 복잡도 제한, 그룹 분할, 별도 배치 평가로 연결한다.

1. **분할을 먼저 수행:** Batch 1을 Train 33셀(17개 프로토콜), Valid 13셀(6개 프로토콜)로 분리한다. `GroupShuffleSplit(test_size=0.25, random_state=42)`의 25%는 프로토콜 그룹 비율이다. 개별 셀 Hold-out만으로 프로토콜 누수를 막을 수 없어 Hold-out과 CV 모두 그룹을 유지한다.
2. **시간과 타깃 누수 차단:** 한 셀의 시계열 시점을 섞지 않고 실제 사이클 10·100만 사용한다. 현재 분할은 완성된 셀 기록 간 그룹 분할이며 시점별 무작위 분할이 아니다. 타깃, 장·단수명 그룹, 최종 관측 길이, 후기 열화 정보를 피처로 사용하지 않는다.
3. **학습 부분에서 피처 사양 결정:** `dq_min = min_V(ΔQ)`와 `log_dq_var = log10(var_V(ΔQ) + 1e-12)`를 계산한다. 각 학습 부분에서 공통 전압 범위와 1,000점 격자를 정하며 검증·Test에는 같은 사양을 적용한다. 범위를 덮지 못하는 곡선은 외삽하지 않고 피처 결측으로 처리한다.
4. **CV 안에서 전처리:** `DeltaQFeatures → SimpleImputer(median) → StandardScaler → Regressor` Pipeline을 사용한다. 결측 대체와 스케일러는 각 fold의 train에서만 fit한다. 두 변수의 단위·척도가 달라 Ridge·Elastic Net·SVR에는 표준화가 필요하다. 정규화는 데이터가 정규분포가 됐다는 의미가 아니다.
5. **복잡도와 평가를 통제:** 무작위성이 있는 모델·분할과 NumPy/Python seed를 42로 고정한다. 결정적인 GroupKFold에는 별도 seed가 필요하지 않다. 소표본에서 불필요한 변수·과도한 탐색·깊은 트리를 늘리지 않고, MAPE로 후보를 비교한다.

### 후보 모델과 EDA 기반 선정 근거

| 후보 모델 | 알고리즘 관점 | 연결되는 데이터 특성과 검증할 가설 | 현재 구현의 복잡도 제어 |
| --- | --- | --- | --- |
| Dummy Regressor | 학습 수명 중앙값을 모든 셀에 예측 | 배치별 수명 수준이 달라 두 피처를 사용한 모델이 단순 기준보다 나은지 확인 | `strategy='median'` |
| Ridge Regression | 선형 회귀에 L2 페널티를 더해 계수 크기를 억제 | 33개 Train 셀과 두 연속 피처에서는 단순한 관계가 안정적일 수 있다. 두 피처가 중복 정보를 가지면 계수 변동을 완화할 수 있다. | 표준화, `alpha=[0.01, 0.1, 1, 10, 100]` |
| Elastic Net | L1·L2 페널티로 계수 축소와 일부 계수 제거 | 두 요약 피처 중 하나의 추가 설명력이 작은지 비교한다. 입력이 두 개뿐이므로 변수 선택의 이점은 제한적일 수 있다. | `alpha=[0.1, 1, 10]`, `l1_ratio=[0.2, 0.8]` |
| RBF-SVR | 커널로 부드러운 비선형 관계를 표현하고 ε 이내 오차를 허용 | ΔQ 크기·퍼짐의 효과가 선형이 아닐 가능성을 검증한다. 곡선 모양만으로 비선형성을 확정하지 않는다. | 표준화, 제한된 C·gamma·epsilon 탐색 |
| Random Forest Regressor | 여러 트리의 예측을 평균해 구간별 관계·상호작용을 표현 | 두 피처 조합에 따른 수명 구간 차이를 포착할 수 있다. 다만 학습에 없는 단수명 영역으로의 외삽은 어렵다. | 최대 깊이 2·4, 최소 리프 셀 수 2·4 |
| Gradient Boosting Regressor | 앞선 오차를 얕은 트리로 순차 보정 | 선형 모델이 놓치는 피처 구간별 효과를 보완할 가능성을 비교한다. 소표본에서는 반복 보정이 과적합을 일으킬 수 있다. | 깊이 1·2, 학습률 0.03·0.1, 트리 수 50·100 |

현재는 두 입력 피처와 수십 개의 셀로 학습하므로 딥러닝의 많은 파라미터를 안정적으로 추정할 근거가 부족하다. 순차 모델을 쓰려면 전체 초기 시계열을 입력으로 정의하고 더 많은 독립 셀을 확보해야 한다. LightGBM은 추가 비교 후보가 될 수 있으나 현재 규모에서 먼저 단순 정규화 회귀와 얕은 트리로 개선 여부를 확인한다. 첨부 예시의 매출 데이터 분포나 Tweedie 목적함수를 이 배터리 회귀에 그대로 적용하지 않는다.

### 모델 선택 절차와 현재 결론

- **Train (Batch 1 CV):** Train 부분의 3-fold Nested Group CV로 비교한다. 바깥 fold는 평가, 안쪽 Group CV는 하이퍼파라미터 튜닝에 사용하며, 전처리도 안쪽 학습 부분에서 fit한다.
- **선택 결과:** 기존 검증 환경에서 Ridge(`alpha=0.1`)가 후보 중 가장 낮은 평균 MAPE 7.89%를 보여 선택됐다. 선형 관계가 진실임을 입증한 결과가 아니라 현재 데이터·분할·탐색 범위에서의 선택이다.
- **Valid (Batch 1 Hold-out):** Train CV로 선택한 모델을 독립 프로토콜 그룹에서 확인한다. Valid가 나쁘다는 이유로 후보를 재선택하지 않는다.
- **Test (Batch 2):** 모델·피처·전처리 사양을 확정한 뒤 마지막에 평가한다. 기존 Test는 모델 재선택·튜닝에 쓰지 않았다. 배치 차이를 본 뒤 추가로 설계하는 개선 모델에는 새로운 미사용 평가 데이터가 필요하다.
- **실무 해석:** Batch 1에서 좋은 모델이라도 Batch 2의 단수명 셀을 과대예측할 수 있다. 따라서 현재 결론은 초기 셀 평가용 기준 모델이며, 현장 배포 성능을 입증한 모델은 아니다. 원논문 MAPE 9.1%는 분할·입력·평가 조건이 완전히 같다고 확인되지 않은 참고값으로 해석한다.

---
## 성능 결과

| 구분                     | MAPE (%) | 비고                                        |
| ------------------------ | -------: | ------------------------------------------- |
| Train (Batch 1 CV)       |     7.89 |                                             |
| Valid (Batch 1 Hold-out) |    12.56 |                                             |
| Test (Batch 2)           |    25.47 |                                             |
| Gap (Train-Valid)        |    -4.66 | (−) : 과적합 의심, Gap 단위 pp              |
| Gap (Valid-Test)         |   -12.91 | (−) : 배치 간 일반화 저하 의심, Gap 단위 pp |
| Gap (Target-Test)        |   -16.37 | Target : 원논문 9.1%, Gap 단위 pp           |

---
## 오류 분석
- 저장된 Test 예측에서 APE가 가장 큰 세 셀은 아래와 같으며 모두 약 400~450사이클의 짧은 수명을 과대예측했다.

| Batch 2 셀 ID | 프로토콜        | 실제 수명 | 예측 수명 | APE (%) |
| ------------- | --------------- | --------: | --------: | ------: |
| 18            | 5.2C(50%)-4.25C |       449 |    733.43 |   63.35 |
| 6             | 3.6C(9%)-5C     |       393 |    641.51 |   63.23 |
| 15            | 3.6C(9%)-5C     |       396 |    634.92 |   60.33 |

- 세 셀 모두 두 특징의 결측 대체가 발생하지 않았다. 따라서 해당 오류를 입력 결측만으로 설명하기 어렵다.
- 원인 가설: Batch 1에 <500사이클 셀이 없고 Batch 2는 단수명 비중이 높아, 학습 범위 밖 수명에 대한 일반화가 어려울 수 있다. 두 ΔQ 특징만으로 충전·실험 조건 차이를 충분히 표현하지 못했을 가능성도 있다.
- 개선 방향: 별도 개발 실험에서 단수명 학습 셀 확보, 초기 충전시간·단계별 C-rate·전환 SOC 추가, EOL 정의·측정 이력 확인을 검토한다. 현재 Test 결과에 맞춘 재튜닝은 수행하지 않았으며, 추가 모델의 최종 평가에는 새로운 미사용 데이터가 필요하다.

---
## 지표 해석과 한계

- MAPE는 작을수록 좋습니다. Gap은 요청대로 **왼쪽−오른쪽**이며, 음수는 오른쪽 오차가 더 큼을 뜻합니다.

- Train−Valid는 CV 평균과 독립 Hold-out의 차이입니다. 일반적인 학습 내 오차와의 차이가 아니므로 차이는 과적합뿐 아니라 검증 프로토콜의 난이도와 소표본 변동도 반영합니다.

- Valid−Test는 Batch 간 일반화 차이, Target−Test는 9.1% 참고값과의 차이(percentage points)입니다.

- Batch 간 조건·수명 분포가 달라 Test 오차가 증가할 수 있습니다. 결과를 보고 Test에
  맞춘 튜닝을 수행하지 않습니다. 이후 추가 실험은 새로운 미사용 평가 데이터가 필요합니다.

- 누락 수명이 EOL 미도달에 따른 중도 종료라면 일반 회귀에서 제외할 때의 선택 편향을 확인해야 합니다. 원본 cycle_life를 사용하며 EOL 정의나 연속 측정 보정은 추측으로 변경하지 않습니다.

- 바코드를 읽지 못한 셀의 물리적 독립성은 파일 기록만으로 완전히 보장할 수 없습니다.

- Pipeline은 전처리 fit 범위를 제한합니다. 이후 실운용에서도 특징 추출 사양을 고정해야 합니다.

---
## ESS 도메인 해석
분석 결과를 실제 ESS 운영 관점에서 해석

- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가?
    현재 모델은 BESS의 자동 제어보다 셀 평가와 점검 우선순위를 정하는 보조 도구로 활용하는 것이 적절합니다.

    - **활용 가능한 의사결정**
      - *셀 입고·선별*: 초기 100사이클 평가 후 예상 수명이 짧은 셀을 추가 검사 대상으로 선정합니다.
  
      - *점검 우선순위*: 수명이 짧게 예측된 셀·모듈의 용량과 온도 등을 먼저 점검하는 참고자료로 활용합니다. 모듈 단위 적용은 별도 검증이 필요합니다.

      - *교체·예비품 계획*: 현장 데이터로 검증한 이후, 교체 수요와 예비품 확보 계획을 지원할 수 있습니다.


- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가?
  
  - **현재 한계**
    - *조건 변화에 취약*: Batch 1 CV MAPE 7.89%보다 Batch 2 오차가 크게 높고, 일부 단수명 셀의 수명을 과대예측했습니다.

    - *수명 표현의 한계*: 예측 대상은 총 Cycle Life입니다. 현재 시점의 잔여수명이나 사용 가능 연수를 직접 예측하지 않습니다.

    - *운영 특성 미반영*: 두 ΔQ 특징만 사용해 온도·SOC·불규칙 충방전과 보관 중 열화를 충분히 반영하지 못합니다. 실제 수명 평가에는 사이클 열화와 달력 열화를 함께 고려해야 합니다. 

    - *측정 제약*: 현장에서 사이클 10·100의 Q(V)를 동일한 조건으로 확보할 수 있는지 확인해야 합니다.
  
  - **실 배포 전 필요한 사항**
    1. *현장 데이터 검증*: 목표 BESS의 셀 종류·운전 조건에서 검증하고, 셀 예측이 모듈·랙 수준에서도 유효한지 확인합니다.

    2. *측정·EOL 기준 통일*: 전압 격자, 용량 측정 조건, 사이클 정의와 수명 종료 기준을 고정합니다.

    3. *불확실성 평가*: 예측 구간을 제공하고, 특히 실제보다 수명을 길게 예측하는 오류를 관리합니다.

    4. *운영 시험*: 먼저 실제 제어에 영향을 주지 않는 방식으로 예측을 기록하고, 점검 결과와 비교합니다.

    5. *BMS와 역할 구분*: 이 모델은 수명 추정용입니다. 과전압·과온 등 기존 보호 기능과 안전 감시는 별도로 유지해야 합니다.

---
## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
- 데이터셋: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle

---
## 팀 구성
- 엄진용 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch1, Batch2)
