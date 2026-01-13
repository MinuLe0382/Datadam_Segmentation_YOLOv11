# Datadam Segmentation YOLOv11

3D Segmentation → Style Transfer 파트 중 3D Segmentation을 수행하기 위한 Segmentation 파트의 코드

기존의 Foundation 모델이 학습하지 않은 네일/페디 부분의 segmentation을 수행하기 위해 YOLOv11을 사용. YOLOv11-seg 모델은 segmentation task에 최적화된 모델로, Backbone(CSPNet), Neck(PANet), Head(Segmentation Head)로 구성됨

![YOLOv11Architecture](YOLOv11_Architecture.png)

## 데이터셋

- **Train**: 32,000 images
- **Validation**: 4,000 images  
- **Test**: 4,000 images

## Classes

- `0`: Fingernail (손톱)
- `1`: Toenail (발톱)


## 📁 프로젝트 구조

```
Datadam_Segmentation_YOLOv11/
├── .devcontainer/          
│   ├── Dockerfile
│   └── devcontainer.json
├── datasets/              
│   ├── train/
│   ├── val/
│   └── test/
├── datasets_lists/       
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── outputs/               
│   ├── runs/             
│   └── prediction_results/ 
├── src/                    
│   ├── logger.py         
│   ├── mask_processing.py 
│   ├── metrics.py         
│   ├── predictor.py       
│   └── visualize.py       
├── weights/               
├── preprocess.py        
├── train.py            
├── eval.py            
├── nail.yaml        
└── requirements.txt    
```


## 입출력 사양

### 입력 (Input)

| 항목 | 사양 |
|------|------|
| 형식 | RGB 이미지(.png) |
| 해상도 | 임의의 이미지 (자동 reshape)|

### 출력 (Output)

| 항목 | 사양 |
|------|------|
| 형식 | 세그멘테이션 마스크(.png) |
| 해상도 | 입력과 동일 |
| 채널 | 1 (이진 마스크) |
| 값 범위 | 0 (배경), 1 (객체) |

```
```
## 기본 학습 설정

### 최적화 (Optimizer)

| 항목 | 설정값 |
|------|--------|
| Optimizer | SGD |
| 초기 학습률 | 0.01 |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |

### 학습 파라미터

| 항목 | 설정값 |
|------|--------|
| Epochs | 50 |
| Batch Size | 9 |
| GPU | 3개 |

### 데이터 증강

| 항목 | 설정값 |
|------|--------|
| 회전 | ±15도 |
| 이동 | ±10% |
| 크기 조절 | ±20% |
| 좌우 반전 | 50% |
| 상하 반전 | 50% |
| HSV (h/s/v) | 0.015 / 0.7 / 0.4 |
```

```
## 학습 결과
test 데이터셋에 대하여 mIoU 0.92달성
```

```
## 라이센스

이 프로젝트는 Apache License 2.0에 따라 배포됩니다.

```
Copyright 2025 광운대학교

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```