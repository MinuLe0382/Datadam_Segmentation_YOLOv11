# src/preprocess.py
"""
라벨 생성 모듈 (txt 파일 기반)
- split.py가 생성한 txt 파일을 읽어 해당 이미지에 대해 라벨 생성
- 심볼릭 링크 생성
- YOLO 라벨 (.txt) 생성
"""

import os
import json
import re
from typing import Dict, List, Optional

# =============================================================================
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# 클래스 매핑: 포즈 번호 → 클래스 ID
POSE_TO_CLASS = {
    '01': 0, '02': 0, '03': 0,  # 손 (fingernail)
    '04': 1                      # 발 (toenail)
}

# =============================================================================
def extract_info_from_filename(filename: str) -> Optional[Dict]:
    """
    파일명에서 정보 추출
    예: P020_01_03_RGB_01.png → person_id: 20, skin_tone: 01, pose: 03
    """
    name = os.path.splitext(filename)[0]
    pattern = r'^P(\d{3})_(\d{2})_(\d{2})_RGB_(\d{2})$'
    match = re.match(pattern, name)
    
    if match:
        return {
            'person_id': int(match.group(1)),
            'skin_tone': match.group(2),
            'pose': match.group(3),
            'img_num': match.group(4)
        }
    return None


def get_class_from_pose(pose: str) -> int:
    # 포즈 번호에서 클래스 ID 반환
    return POSE_TO_CLASS.get(pose, 0)


def get_metadata_path_for_image(image_path: str, metadata_root: str) -> Optional[str]:
    """
    이미지 경로에서 해당 메타데이터 JSON 경로 추출
    
    이미지: .../피부톤1/손/P001_01_02_RGB_05.png
    메타데이터: {metadata_root}/피부톤1/손/P001_01_02_metadata.json
    """
    filename = os.path.basename(image_path)
    info = extract_info_from_filename(filename)
    if not info:
        return None
    
    # 피부톤 폴더 추출
    parts = image_path.replace('\\', '/').split('/')
    skin_tone = None
    body_part = None
    
    for i, part in enumerate(parts):
        if part.startswith('피부톤'):
            skin_tone = part
            if i + 1 < len(parts) and parts[i + 1] in ['손', '발']:
                body_part = parts[i + 1]
            break
    
    if not skin_tone or not body_part:
        return None
    
    # 메타데이터 파일명: P{person}_{skin}_{pose}_metadata.json
    person_str = f"P{info['person_id']:03d}"
    metadata_filename = f"{person_str}_{info['skin_tone']}_{info['pose']}_metadata.json"
    
    return os.path.join(metadata_root, skin_tone, body_part, metadata_filename)


def parse_metadata_json(json_path: str) -> Dict:
    # 메타데이터 JSON 파싱하여 이미지별 폴리곤 추출
    if not os.path.exists(json_path):
        return {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result = {}
    rgb_size = data.get('related_data', {}).get('rgb_size', [2124, 2832])
    
    for item in data.get('image_meta_info', []):
        image_id = item.get('image_id')
        polygons = item.get('polygon', [])
        
        if image_id and polygons:
            result[image_id] = {
                'polygons': polygons,
                'width': rgb_size[0],
                'height': rgb_size[1]
            }
    
    return result


def polygon_to_yolo(polygon: List[float], img_width: int, img_height: int, class_id: int) -> str:
    # 폴리곤 좌표를 YOLO 형식으로 변환
    normalized = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / img_width
        y = polygon[i + 1] / img_height
        normalized.extend([x, y])
    
    coords_str = " ".join(f"{c:.6f}" for c in normalized)
    return f"{class_id} {coords_str}"


def create_symlinks_for_list(image_paths: List[str], target_dir: str) -> int:
    # 이미지 경로 리스트에 대해 심볼릭 링크 생성
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    
    for src_path in image_paths:
        filename = os.path.basename(src_path)
        link_path = os.path.join(target_dir, filename)
        
        if os.path.exists(link_path) or os.path.islink(link_path):
            count += 1
            continue
        
        abs_src = os.path.abspath(src_path)
        if os.path.exists(abs_src):
            os.symlink(abs_src, link_path)
            count += 1
    
    return count


def create_label_for_image(
    image_path: str, 
    metadata_root: str, 
    labels_dir: str
) -> bool:
    # 단일 이미지에 대해 YOLO 라벨 생성
    filename = os.path.basename(image_path)
    info = extract_info_from_filename(filename)
    if not info:
        return False
    
    image_id = os.path.splitext(filename)[0]
    label_path = os.path.join(labels_dir, f"{image_id}.txt")
    
    # 이미 존재하면 스킵
    if os.path.exists(label_path):
        return True
    
    # 메타데이터 찾기
    metadata_path = get_metadata_path_for_image(image_path, metadata_root)
    if not metadata_path or not os.path.exists(metadata_path):
        return False
    
    # 메타데이터 파싱
    parsed = parse_metadata_json(metadata_path)
    if image_id not in parsed:
        return False
    
    img_info = parsed[image_id]
    class_id = get_class_from_pose(info['pose'])
    
    # 라벨 파일 작성
    os.makedirs(labels_dir, exist_ok=True)
    with open(label_path, 'w') as f:
        for polygon in img_info['polygons']:
            yolo_line = polygon_to_yolo(polygon, img_info['width'], img_info['height'], class_id)
            f.write(yolo_line + "\n")
    
    return True


def prepare_data_from_list(
    data_list_path: str,
    source_root: str,
    images_dir: str = "datasets/images",
    labels_dir: str = "datasets/labels"
) -> bool:
    """
    txt 파일 기반으로 데이터 준비 (심볼릭 링크 + 라벨 생성)
    
    Args:
        data_list_path: 이미지 경로 리스트 파일 (train.txt, test.txt 등)
        source_root: 소스 데이터 루트 (datasets/)
        images_dir: 심볼릭 링크 디렉토리
        labels_dir: 라벨 디렉토리
        
    Returns:
        bool: 성공 여부
    """
    if not os.path.exists(data_list_path):
        print(f"오류: 데이터 리스트 파일을 찾을 수 없습니다: {data_list_path}")
        print("먼저 split.py를 실행하세요: python split.py --source datasets")
        return False
    
    # 이미지 경로 읽기
    with open(data_list_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    if not image_paths:
        print(f"오류: 빈 데이터 리스트: {data_list_path}")
        return False
    
    print(f"데이터 준비 시작: {len(image_paths)}개 이미지")
    
    # 메타데이터 루트 경로
    metadata_root = os.path.join(source_root, "3.Other", "메타데이터", "원천데이터1. 메타데이터")
    
    # 1. 심볼릭 링크 생성
    print(f"  [1/2] 심볼릭 링크 생성...")
    symlink_count = create_symlinks_for_list(image_paths, images_dir)
    print(f"        완료: {symlink_count}개")
    
    # 2. 라벨 생성
    print(f"  [2/2] YOLO 라벨 생성...")
    label_count = 0
    for path in image_paths:
        if create_label_for_image(path, metadata_root, labels_dir):
            label_count += 1
    print(f"        완료: {label_count}개")
    
    print(f"데이터 준비 완료")
    return True
