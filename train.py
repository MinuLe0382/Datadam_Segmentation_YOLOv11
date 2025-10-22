import os
from ultralytics import YOLO

# path settings
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # train.py 파일이 있는 프로젝트 루트 디렉터리
yaml_file = 'datasets/nail.yaml' # yaml 파일이 있는 디렉터리
weights_dir = os.path.join(ROOT_DIR, 'weights')
os.makedirs(weights_dir, exist_ok=True)
output_project_dir = os.path.join(ROOT_DIR, 'outputs', 'runs')
model_path = os.path.join(weights_dir, 'yolo11l-seg.pt')

# Load a model
model = YOLO(model_path)  # load a pretrained model (recommended for training)

results = model.train(data=yaml_file, 
                      project=output_project_dir,
                      # imgsz: 32의 배수
                      # device: 사용가능한 GPU 넘
                      epochs=3, imgsz=1632, device=[1, 2], batch=8,
                      degrees=15.0,      # 이미지 회전 각도 (±15도)
                      translate=0.1,   # 이미지 이동 비율 (±10%)
                      scale=0.2,       # 이미지 크기 조절/확대 비율 (±20%)
                      fliplr=0.5,      # 50% 확률로 좌우 반전
                      flipud=0.5,      # 50% 확률로 상하 반전
                      hsv_h=0.015,     # 색상(Hue) 변형 강도
                      hsv_s=0.7,       # 채도(Saturation) 변형 강도
                      hsv_v=0.4        # 명도(Value) 변형 강도
)