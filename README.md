# ESS 배터리 수명 예측 
목적 작성 


## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12)
- 평가 데이터 : Batch 2 (2018-02-20)
- 태스크 : Regression (Cycle Life 예측)


## 파일 구조 (sample) 
```
├── data/
│   └── README.md          
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── train.py
├── results/
│   └── model_performance.csv
├── requirements.txt
└── README.md
```


## 환경 설정 (sample) 
```bash
git clone https://github.com/eom-skala/DS-Mini_Design
cd ess-battery-project
pip install -r requirements.txt
```


## EDA 

- Cycle Life 분포
	- 분포 형태 및 장단수명 비율 요약
	- 핵심 발견 : (팀이 발견한 인사이트를 한 줄로)

- 열화 곡선 분석
	- 장수명 vs 단수명 셀의 열화 속도 차이
	- Knee point 존재 여부 및 발생 시점
	- 핵심 발견 :

- ΔQ(V) 곡선 분석
	- Cycle 100 - Cycle 10 차이 곡선 형태
	- 장단수명 셀 간 ΔQ 형태 비교
	- 핵심 발견 :

- 충전 속도(C-rate)와 수명의 관계
	- 충전 프로토콜별 평균 수명 비교 결과
	- 핵심 발견 :

- (추가 확인한 내용 작성) 


## Modeling 

### 피처 엔지니어링 전략
EDA 결과를 바탕으로 선택한 피처와 그 근거를 기술


### 모델 선택 및 근거
- 후보 모델 : Dummy Regressor, Ridge Regression, Elastic Net, RBF-SVR, RandomForest Regressor, Gradient Boosting Regressor
- 최종 모델 :
- 선택 이유 :


## 성능 결과
Format에 맞춰 작성


## 오류 분석
- 모델이 가장 크게 틀린 셀의 공통점
- 원인 가설 및 개선 방향

## 지표 해석과 한계

- MAPE는 작을수록 좋습니다. Gap은 요청대로 **왼쪽−오른쪽**이며, 음수는 오른쪽 오차가 더 큼을 뜻합니다.
- Train−Valid는 CV 평균과 독립 Hold-out의 차이입니다. 일반적인 학습 내 오차와의 차이가 아니므로
  차이는 과적합뿐 아니라 검증 프로토콜의 난이도와 소표본 변동도 반영합니다.
- Valid−Test는 Batch 간 일반화 차이, Target−Test는 9.1% 참고값과의 차이(percentage points)입니다.
- Batch 간 조건·수명 분포가 달라 Test 오차가 증가할 수 있습니다. 결과를 보고 Test에
  맞춘 튜닝을 수행하지 않습니다. 이후 추가 실험은 새로운 미사용 평가 데이터가 필요합니다.
- 누락 수명이 EOL 미도달에 따른 중도 종료라면 일반 회귀에서 제외할 때의 선택 편향을 확인해야 합니다.
  원본 cycle_life를 사용하며 EOL 정의나 연속 측정 보정은 추측으로 변경하지 않습니다.
- 바코드를 읽지 못한 셀의 물리적 독립성은 파일 기록만으로 완전히 보장할 수 없습니다.
- Pipeline은 전처리 fit 범위를 제한합니다. 이후 실운용에서도 특징 추출 사양을 고정해야 합니다.


## ESS 도메인 해석
분석 결과를 실제 ESS 운영 관점에서 해석

- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가?
- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가?


## 참고문헌
- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
- https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle


## 팀 구성
- 엄진용 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch1, Batch2)
