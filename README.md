# Datadam Segmentation YOLOv11

한국지능정보사회진흥원 2025년도 초거대AI 확산 생태계 조성 사업 [과제 3]

컨소시엄의 제안사항 3D Segmentation → Style Transfer 파트 중 3D Segmentation을 수행하기 위한 Segmentation 파트의 코드

기존의 Foundation 모델이 학습하지 않은 네일/페디 부분의 segmentation을 수행하기 위해 YOLOv11을 사용

## 📊 Dataset

- **Train**: ~32,000 images
- **Validation**: ~4,000 images  
- **Test**: ~4,000 images

## 🎯 Classes

- `0`: Fingernail (손톱)
- `1`: Toenail (발톱)

## 🚀 Installation

프로젝트 설정 시 Dev Container를 통한 구축을 위한 Dockerfile 제공. 로컬 수동 설치도 가능

### Option 1: Dev Container (Recommended)

#### Prerequisites
- Visual Studio Code (with Dev Containers extension)
- Docker Desktop

#### Steps
1. Clone Repository
```bash
git clone https://github.com/MinuLe0382/Datadam_Segmentation_YOLOv11.git
cd Datadam_Segmentation_YOLOv11
```

2. Open in Visual Studio Code
```bash
code .
```

3. Reopen in Container
   - Press `F1` or `Ctrl+Shift+P`
   - Select `Dev Containers: Reopen in Container`
   - Wait for container to build and start

### Option 2: Manual Local Installation

#### Prerequisites
- Python 3.8+
- CUDA 12.1 (for GPU support)

#### Steps
1. Clone Repository
```bash
git clone https://github.com/MinuLe0382/Datadam_Segmentation_YOLOv11.git
cd Datadam_Segmentation_YOLOv11
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Datadam_Segmentation_YOLOv11/
├── .devcontainer/          # Dev container configuration
│   ├── Dockerfile
│   └── devcontainer.json
├── datasets/               # Training/validation/test data
│   ├── train/
│   ├── val/
│   └── test/
├── datasets_lists/         # Image path lists (generated)
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── outputs/                # Training and evaluation outputs
│   ├── runs/              # Training runs
│   └── prediction_results/ # Evaluation results
├── src/                    # Core modules
│   ├── logger.py          # Evaluation logging
│   ├── mask_processing.py # Mask processing utilities
│   ├── metrics.py         # IoU calculation
│   ├── predictor.py       # TTA prediction
│   └── visualize.py       # Visualization utilities
├── weights/                # Model weights
├── preprocess.py          # Dataset preprocessing
├── train.py               # Training script
├── eval.py                # Evaluation script
├── nail.yaml              # Dataset configuration
└── requirements.txt       # Python dependencies
```

## 🔧 Usage

> [!IMPORTANT]
> Before training or evaluation, you **must** run the preprocessing script to convert COCO annotations to YOLO format and generate dataset lists.

### 1. Data Preprocessing

Convert COCO format annotations to YOLO format and generate image lists:

```bash
python preprocess.py
```

This script will:
- Convert COCO JSON annotations to YOLO txt format
- Generate image path lists in `datasets_lists/`
- Create labels in each dataset folder

### 2. Training

Train the YOLOv11 segmentation model:

```bash
# Basic training
python train.py --epochs 50 --device 0 --batch 8

# Multi-GPU training
python train.py --epochs 50 --device 0 1 2 --batch 9

# Custom configuration
python train.py --epochs 100 --device 0 1 --batch 16
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 50)
- `--device`: GPU device IDs (default: 0)
- `--batch`: Batch size (default: 8)

**Training Features:**
- Image size: 1632×1632 (32의 배수)
- Data augmentation:
  - Rotation: ±15°
  - Translation: ±10%
  - Scale: ±20%
  - Horizontal flip: 50%
  - Vertical flip: 50%
  - HSV augmentation

### 3. Evaluation

Evaluate trained model on test set:

```bash
# Full evaluation with TTA
python eval.py \
  --model_path outputs/runs/train/weights/best.pt \
  --data_list datasets_lists/test.txt

# Evaluation without TTA
python eval.py \
  --model_path outputs/runs/train/weights/best.pt \
  --data_list datasets_lists/test.txt \
  --no-aug

# Fast evaluation (no visualization)
python eval.py \
  --model_path outputs/runs/train/weights/best.pt \
  --data_list datasets_lists/test.txt \
  --no-save-masks \
  --no-low-iou-vis
```

**Arguments:**
- `--model_path`: Path to trained model (.pt file) [Required]
- `--data_list`: Path to dataset list txt file [Required]
- `--no-save-masks`: Do not save predicted masks
- `--no-low-iou-vis`: Do not save low IoU visualizations
- `--low_iou_threshold`: IoU threshold for visualization (default: 0.01)
- `--no-aug`: Disable Test-Time Augmentation

**Evaluation Outputs:**
- `prediction/`: Predicted masks
- `vis/`: Low IoU comparison visualizations
- `evaluation_results.json`: Detailed metrics
- `iou_distribution.png`: IoU score distribution

## 🧪 Test-Time Augmentation (TTA)

The evaluation script supports TTA with:
- Multi-scale testing: [0.8, 0.9, 1.0, 1.1, 1.2]
- Horizontal flip augmentation
- Ensemble averaging for robust predictions

## 📊 Evaluation Metrics

- **mIoU (mean Intersection over Union)**: Primary metric
- **Trimap-based IoU**: Excludes boundary pixels for fair evaluation
- **Per-image IoU scores**: Detailed analysis
- **Class-wise predictions**: Fingernail vs Toenail detection

## 🛠️ Key Features

1. **Modular Architecture**: Clean separation of concerns in `src/` modules
2. **Comprehensive Logging**: JSON-based evaluation results with metadata
3. **Visualization Tools**: Automatic generation of comparison figures
4. **Flexible Preprocessing**: COCO to YOLO format conversion
5. **Advanced Augmentation**: Both training-time and test-time augmentation
6. **GPU Support**: Multi-GPU training capability

## 📦 Dependencies

Main dependencies (see `requirements.txt` for full list):
- PyTorch 2.1.1 (CUDA 12.1)
- Ultralytics YOLOv11
- OpenCV 4.8.1
- NumPy 1.26.4
- Matplotlib
- MONAI 1.3.0

## 📝 Quick Start Tutorial

```bash
# 1. Preprocess dataset
python preprocess.py

# 2. Train model
python train.py --epochs 50 --device 0 1 2 --batch 9

# 3. Evaluate model
python eval.py \
  --model_path outputs/runs/train/weights/best.pt \
  --data_list datasets_lists/test.txt \
  --low_iou_threshold 0.5
```

## 📄 License

This project is part of the 2025 Korea Intelligence & Information Society Agency (NIA) Hyperscale AI Ecosystem Development Project.

## 🔗 Repository

[https://github.com/MinuLe0382/Datadam_Segmentation_YOLOv11](https://github.com/MinuLe0382/Datadam_Segmentation_YOLOv11)