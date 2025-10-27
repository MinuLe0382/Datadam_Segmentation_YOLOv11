# Datadam_Segmentation_YOLOv11
한국지능정보사회진흥원 2025년도 초거대AI 확산 생태계 조성 사업 [과제 3 LMM]

컨소시엄의 제안사항 3D Segmentation -> Style Transfer파트 중 3D Segmentation을 수행하기 위한 Segmentation 파트의 코드

기존의 Foundation 모델이 학습하지 않은 네일/페디부분의 segmentation을 수행하기 위해 YOLOv11을 사용

Train 데이터로 약 10,000장, Test 데이터로 1,500장 Validation 데이터로 1,000장 사용
<br />

## Installation

프로젝트 설정시에 Dev Contationer를 구축을 위한 Dockerfile 제공. 혹은 로컬 수동 설치도 가능

### Option 1: Dev Container (Recommended)

### Prerequisites
* Visual Studio Code (with Dev Containers ADD-ONS)
* Docker Desktop

### Steps
1. Clone Repository
```sh
git clone https://github.com/MinuLe0382/Datadam_Segmentation_YOLOv11.git
cd [YOUR_PROJECT_DIRECTORY]
```
2. Open in Visual Studio Code
```sh
code .
```
3. Reopen in Container

### Option 2: Manual Local Installation

### Prerequisites
* Visual Studio Code (with Dev Containers ADD-ONS)
* Docker Desktop

### Steps
1. Clone Repository
```sh
git clone https://github.com/MinuLe0382/Datadam_Segmentation_YOLOv11.git
cd [YOUR_PROJECT_DIRECTORY]
```
2. Open in VSCODE
```sh
code .
```
3. Reopen in Container

