# 사진을 통한 특정지역 판별 프로그램
Tensorflow의 CNN을 활용하여, 학습한 지역을 판별하는 프로그램이다.
총 4인이 개발 프로젝트에 참여하였다.

kears를 이용한 이미지 분류를 주된 목표로 삼았으며 그 중 다수의 지역 사진을 통한 학습 후 특정 지역을 판별 할 수 있는 기능을 구현한다.
딥러닝의 한 종류인 CNN을 이용하여 이미지를 학습하고 테스트 데이터를 통해 검증 한다. 

프로젝트 회의록
--------------------

![image](https://user-images.githubusercontent.com/56360477/135051967-902baf56-e0d6-421f-ab27-3b2fede3ed82.png)

프로그램 흐름 및 신경망 도식화
-------------------------
![image](https://user-images.githubusercontent.com/56360477/135052061-274fa78f-882b-47f1-9ab3-eb0b0d6261e6.png)

![image](https://user-images.githubusercontent.com/56360477/135052106-ca4b1f36-37e7-4618-8220-4493820240cc.png)


코드 설명
---------------
원활한 학습과, 테스팅을 위하여 프로그램코드를 단계별로 세가지로 분류하였다. 첫번째 코드인 ‘1_데이터 전처리.py’에서는 학습데이터를 로드해, 이미지를 Resize,Labeling하여 npy파일일로 저장한다. 두번째 코드인 ‘2_분류학습.py’에서는 분류가 완료된 학습데이터들을 불러와 CNN모델을 사용하여 학습시킨다. 마지막 코드인 ‘3_테스팅.py’에서는 학습이 완료된 모델에 새로운 테스트 데이터들을 입력시켜, 그 결과 예측값을 출력시킨다. 프로그램은 ‘1_데이터전처리.py’, ’2_분류학습.py’, ‘3_테스팅.py’ 순으로 실행한다.

실행 결과
--------------
- 기준: 학습데이터 (8000장), 그외 데이터(6000장), 테스트 데이터 (29장)

![image](https://user-images.githubusercontent.com/56360477/135052360-0777930c-4f34-4088-b936-25fcd877871c.png)


참고자료
---------------
꿈 많은 사람의 이야기-케라스
(https://lsjsj92.tistory.com/355)

자신만의 이미지 데이터로 CNN 적용해보기
(https://twinw.tistory.com/252)

딥러닝 (6) - CNN (Convolutional Neural Network)
(https://davinci-ai.tistory.com/29)

CNN 구조 - ResNet
(http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=221259295035)

Codetorial-고양이와 개 이미지 분류하기
(https://codetorial.net/tensorflow/classifying_the_cats_and_dogs.html)

Techiepedia-Binary Image classifier CNN using TensorFlow
(https://medium.com/techiepedia/binary-image-classifier-cnn-using-tensorflow-a3f5d6746697)


김동근,『텐서플로 딥러닝 프로그래밍』,가메출판사,2020.09.01,STEP22_02,STEP30_01,Chapter09(CNN),등

