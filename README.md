# MRC_Dataset_BIas-Probe
💡 [AIHUB OPENLAB] 교차성능평가와 적대적 학습을 이용한 기계독해 데이터셋 편향성 분석 (논문 O)

```bash
├── Final_files
│   ├── data_augmentation.py
│   ├── data_preprocessing.py
│   ├── generate_queries.py
│   ├── krx_api.py
│   ├── krx_db_utils.py
│   ├── krx_def.py
│   ├── masking_utils.py
│   └── squad_utils.py
└── Masking-klue-Roberta
    ├── main.py
    ├── multimasking
    │   └── multimasking.py
    ├── singlemaskingforwordtoken
    └── └── singlemasking.py
``` 
---

## * Final_files : 프로젝트 최종 정리 파일

1. data_augmentation: 질문(query)의 평서문화, 질문 및 답변 내 명사 추출, 형제어 추출

2. generate_queries: 추출된 형제어를 통해서 적대적 문장 생성 및 적대적 문장을 포함한 학습 및 평가 데이터 생성

3. krx_*: KorLex 2.0 API 실행을 위한 클래스 구현

4. masking_utils: Mask Language Model을 이용하여 추출된 형제어들의 확률 계산 및 확률이 높은 형제어 선택

5. squad_utils: 기계독해 데이터 생성에 필요한 함수 모음(KorQuAD에서 제공되는 코드를 수정)

---

## * Masking-klue-Roberta : 프로젝트 진행 과정 중 고안한 두 가지 마스킹 방식 


### 1) Multi-masking

멀티마스킹은 교체될 단어 A 전체를 [MASK] 하고 모델에서 새로 들어갈 단어 B의 토큰 개수를 기준으로 앞에서부터 차례로 한 토큰씩 맞춰 더해가면서 예측하는 클래스입니다.

#### 예)
나는 대한민국에 산다. : [대][한][민][국] -> [미][국]

나는 [Mask][Mask]에 산다. (0.001) -> 나는 미[Mask]에 산다. (0.004) -> 나는 미국에 산다. ==> 미국's Probability : (0.001 + 0.004) /2)


### 2) Single-masking

싱글마스킹은 교체될 단어 A가 아닌 새로 들어갈 단어 B에 초점을 맞추어 B를 토큰화한 후 차례로 토큰 중 하나씩만 [MASK]를 한 후 그때그때 적절한 토큰을 예측하는 클래스입니다.

#### 예)
나는 대한민국에 산다. : [대][한][민][국] -> [미][국]

나는 미[Mask]에 산다. (0.005) -> 나는 [Mask]국에 산다. (0.0012) -> 나는 미국에 산다. ==> 미국's Probability : (0.005 + 0.0012) /2)

---

## * 적대적 학습을 위한 질문 내 명사 -> 형제어 치환 메커니즘

![image](https://user-images.githubusercontent.com/64192139/209921910-feabd184-d60c-4def-9b26-c43dd5ece750.png)



* *본 프로젝트는 [2022 AI 데이터 품질 개선 오픈랩 프로그램]을 통해서 진행되었습니다.*
* 
