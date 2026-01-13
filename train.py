import os
from ultralytics import YOLO
import argparse

from src.preprocess import prepare_data_from_list

# =============================================================================
parser = argparse.ArgumentParser(description='YOLO Segmentation Training Script')

parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs (default: 50)')

parser.add_argument('--device', type=int, nargs='+', default=[0],
                    help='GPU device IDs (default: 0)')

parser.add_argument('--batch', type=int, default=8,
                    help='Batch size (default: 8)')

parser.add_argument('--source', type=str, default='datasets',
                    help='소스 데이터 루트 (default: datasets)')

parser.add_argument('--data-list', type=str, default='datasets_lists/train.txt',
                    help='학습 데이터 리스트 (default: datasets_lists/train.txt)')

args = parser.parse_args()

# =============================================================================
# 데이터 준비 (심볼릭 링크 + 라벨 생성)
# =============================================================================
IMAGES_DIR = os.path.join(args.source, "images")
LABELS_DIR = os.path.join(args.source, "labels")

print("데이터 준비 중...")
success = prepare_data_from_list(
    data_list_path=args.data_list,
    source_root=args.source,
    images_dir=IMAGES_DIR,
    labels_dir=LABELS_DIR
)
if not success:
    print("데이터 준비 실패. 먼저 split.py를 실행하세요.")
    exit(1)

# =============================================================================
# 학습 설정
# =============================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_file = 'nail.yaml'
weights_dir = os.path.join(ROOT_DIR, 'weights')
os.makedirs(weights_dir, exist_ok=True)
output_project_dir = os.path.join(ROOT_DIR, 'outputs', 'runs')
model_path = os.path.join(weights_dir, 'yolo11l-seg.pt')

model = YOLO(model_path)

print(f"--- Training Configuration ---")
print(f"Epochs: {args.epochs}")
print(f"Device: {args.device}")
print(f"Batch Size: {args.batch}")
print(f"Data List: {args.data_list}")
print(f"-------------------------------")

# =============================================================================
# 학습 실행
# =============================================================================
results = model.train(
    data=yaml_file, 
    project=output_project_dir,
    epochs=args.epochs,
    imgsz=1632,
    device=args.device,
    batch=args.batch,
    degrees=15.0,
    translate=0.1,
    scale=0.2,
    fliplr=0.5,
    flipud=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
)
