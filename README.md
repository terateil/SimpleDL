# SimpleDL
<p>
<img src=https://user-images.githubusercontent.com/99002885/156101338-ee5aba26-23e6-4850-9f6b-4661c5e41950.png width = 100>
</p>

코드 없이 간단한 신경망을 설계, 훈련, 테스트할 수 있는 프로그램입니다.  
Language: python  
GUI: Tkinter/
DL framework: TensorFlow  
Project in December, 2021.

## 시작하기
프로그램을 사용하려면 repository를 clone 하고, `dist/main/main.exe`를 실행시키세요.  
python 코드를 확인하려면 `src/main.py`를 참고하세요.

## 프로젝트의 목적
 1. 코드를 작성하지 않고 목적에 따라 원하는 딥러닝 모델을 설계할 수 있도록 함.
 2. 사용자 친화적이고 직관적인 UI를 통해 비전문가도 쉽게 딥러닝 모델을 다룰 수 있게 함.
 3. 모델 생성, 학습, 테스트, 예측의 기능을 통해 소프트웨어 하나만으로도 사용자가 원하는 대부분의 기능을 수행할 수 있도록 함.

## 사용 방법 - 모델 설계
### 모델 기본정보 입력
프로그램을 실행하고 '**새 모델 생성**'을 선택하고 다음 순서를 따르세요. 
#### 1. 모델 목적 선택
모델로 해결하고자 하는 문제를 선택하세요. 
- 회귀
- 분류
- 그 외(직접 설계)
#### 2. 입력 데이터 정보 입력
모델의 입력 데이터의 정보를 입력하세요.  
- 일반(ex 보스턴 집값 데이터셋) - 특성 개수 입력  
- 이미지(ex mnist 데이터셋) - 이미지 가로 픽셀수, 세로 픽셀수, 채널(흑백, 컬러)  입력  
- 시계열(ex 날씨 데이터셋) - 특성 개수 입력  
#### 3. 타겟 데이터 정보 입력
모델 목적에 따라 적합한 정보를 입력하세요.
- 회귀 - 1로 고정
- 분류 - 분류할 클래스 개수 입력
- 그 외 - 이후에 직접 설계
<p align = "center">
<img src = https://user-images.githubusercontent.com/99002885/156099333-09f2f8e2-aa75-408e-a18d-3c54e6448662.PNG width = 500>
</p>
<p align = "center">
ex) MNIST 데이터셋에 대한 분류 모델 설계 과정
</p>

### 모델 설계
이전 단계의 정보를 바탕으로 모델 베이스가 생성됩니다. 모델 베이스를 바탕으로 원하는 대로 모델을 확장해 보세요.  
Dense(완전연결 층), Conv2D(2차원 합성곱 층), RNN(순환 신경망) 중에서 레이어를 선택할 수 있습니다.  
레이어를 우클릭하여 레이어 삭제, 추가 작업을 할 수 있습니다.  
단, 아래의 기본적인 규칙은 지켜져야 모델이 오류 없이 생성되므로 주의하세요.  

**기본 규칙**
- 일반 입력: Dense 층으로만 구성
- 이미지 입력: 먼저 Conv2D 층을 쌓아 나가고 마지막 Conv2D 층만 ‘데이터 펼치기’ 옵션 반드시 체크. 이후 Dense 층에 연결
- 시계열 입력: 먼저 RNN 층을 쌓아 나가고 마지막 RNN 층을 제외하고 모두 ‘timestep 유지’ 옵션 반드시 체크. 이후 Dense층에 연결
<p align = "center">
<img src = https://user-images.githubusercontent.com/99002885/156102838-fc96cec0-3e4e-4c42-bb25-86f7846cc5ba.png>
</p>
<p align = "center">
ex) 모델 세부 설계 예시
</p>

## 사용 방법 - 모델 훈련, 테스트 및 예측
메인 화면에서 ‘모델 불러오기’를 선택하고, 원하는 모델 파일(`xxx.h5`)를 불러옵니다.  
### 모델 훈련
1. 모드를 **훈련**으로 선택합니다.
2. 입력, 타겟 데이터를 ‘찾아보기’ 버튼을 눌러 불러옵니다.
3. 손실함수, 평가지표, 배치 크기, 학습 횟수를 선택합니다.
4. 모델은 학습이 종료된 시간을 모델명에 붙여 자동 저장됩니다.
### 모델 테스트
모드를 **테스트**로 선택하고, 데이터를 불러옵니다. 이후 배치 크기를 지정해 모델 훈련과 동일하게 진행하세요.
### 예측
학습시킨 모델을 이용해 원하는 결과값을 얻어내는 과정입니다. 결과는 넘파이 파일로 저장할 수 있습니다.

<p align = "center">
<img src = https://user-images.githubusercontent.com/99002885/156104037-dc2c1df2-231a-4125-8888-882ac325f10b.png width = 800>
</p>
<p align = "center">
ex) MNIST 데이터셋에서 모델 훈련, 테스트, 예측을 진행한 모습
</p>

위 모든 과정은 창 하단의 로그에 표시됩니다.

## 추가 정보
### 포함된 샘플 데이터셋 정보
- boston: keras - Boston Housing price regression dataset
- mnist: keras - MNIST digits classification dataset
- climate: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip

더 자세한 정보를 얻고 싶으면 시연영상을 참고하세요.
https://drive.google.com/drive/folders/1aw8e8dY1in7huqBwnfUAWegX9YSmlJpJ?usp=sharing


