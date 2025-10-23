import os
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='YOLO Segmentation Training Script')

parser.add_argument('--epochs', 
                    type=int, 
                    default=50, 
                    help='Number of training epochs (default: 50)')

parser.add_argument('--device', 
                    type=int, 
                    nargs='+',
                    default=[0], 
                    help='GPU device IDs (default: 0)')

parser.add_argument('--batch', 
                    type=int, 
                    default=8, 
                    help='Batch size (default: 8)')

args = parser.parse_args()

# path settings
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # train.py 파일이 있는 프로젝트 루트 디렉터리
yaml_file = 'datasets/nail.yaml' # yaml 파일이 있는 디렉터리
weights_dir = os.path.join(ROOT_DIR, 'weights')
os.makedirs(weights_dir, exist_ok=True)
output_project_dir = os.path.join(ROOT_DIR, 'outputs', 'runs')
model_path = os.path.join(weights_dir, 'yolo11l-seg.pt')

# Load a model
model = YOLO(model_path)  # load a pretrained model

# 파싱된 인자(args)를 사용하여 학습 설정값 출력
print(f"--- Training Configuration ---")
print(f"Epochs: {args.epochs}")
print(f"Device: {args.device}")
print(f"Batch Size: {args.batch}")
print(f"-------------------------------")


results = model.train(data=yaml_file, 
                      project=output_project_dir,
                      epochs=args.epochs,
                      imgsz=1632,       # imgsz: 32의 배수
                      device=args.device, # device: 사용가능한 GPU 넘버
                      batch=args.batch,
                      degrees=15.0,      # 이미지 회전 각도 (±15도)
                      translate=0.1,   # 이미지 이동 비율 (±10%)
                      scale=0.2,       # 이미지 크기 조절/확대 비율 (±20%)
                      fliplr=0.5,      # 50% 확률로 좌우 반전
                      flipud=0.5,      # 50% 확률로 상하 반전
                      hsv_h=0.015,     # 색상(Hue) 변형 강도
                      hsv_s=0.7,       # 채도(Saturation) 변형 강도
                      hsv_v=0.4        # 명도(Value) 변형 강도
)