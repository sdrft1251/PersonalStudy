# Auto Encoder 기반 모델
Anomaly Detection을 목적으로 하며, AE 모델을 활용.  
AE 모델의 경우 Unsupervised learning이 가능하여, 이 방식의 학습을 목표로 함.  
ECG 데이터를 통해서 비정상적인 신호를 잡는 것을 목적으로 함.  
1. Label이 없음.

## 두가지의 모델 형태를 기반으로 개발
1. AE 기반
2. VAE 기반

## Dir Detail
### /model
모델 파일 모음
1. AE_TF_COMPRESS -> AutoEncoder + Transformer + 데이터 압축
2. AE_TF -> AutoEncoder + Transformer
3. DA_RNN -> DualStageAttention + Recurrent모델
4. FFT_AE -> AutoEncoder + FastFourierTransform
5. VAE_VGG16_GRU -> VariationalAutoEncoder + 데이터압축(VGG16) + GRU
6. VRAE -> VariationalRecurrentAutoEncoder
### /train
훈련 관련 함수 -> VAE와 AE를 나누어 파일 만듬  
분산 훈련 함수 추가
### /utils
util 관련 함수들
1. attention 관련 모듈
2. 압축 관련 모듈 -> 효율적인 훈련을 목적으로 활용
3. embedding 관련 모듈
4. data 관련 모듈 -> 데이터 전처리(신호 전처리 + 포맷)
5. report 관련 모듈 -> 결과 올릴 report 관련 모듈
6. streamlit 관련 모듈 -> streamlit lib 사용 예제 모듈
7. loss 관련 모듈 -> 각종 Loss function
### /assist
훈련 보조용 코드