# split.py
"""
데이터 분할 스크립트
- 소스 데이터 스캔
- train.txt, val.txt, test.txt 생성

모드:
  1. 랜덤 분할: --ratio 옵션 사용
  2. 가이드 분할: --guide 옵션 사용 (data_guide/ 파일 기반)
"""

import os
import re
import random
import shutil
import argparse
from typing import List, Set, Tuple, Optional, Dict

# =============================================================================
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
DEFAULT_SPLIT_RATIO = (0.8, 0.1, 0.1)
DEFAULT_SEED = 42

# =============================================================================
# 유틸리티 함수
# =============================================================================
def extract_info_from_filename(filename: str) -> Optional[Dict]:
    """
    파일명에서 정보 추출
    예: P020_01_03_RGB_01.png → person_id: 20, skin_tone: 01, pose: 03, img_num: 01
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


def get_skin_tone_folder(skin_code: str) -> str:
    """피부톤 코드 → 폴더명 (01 → 피부톤1)"""
    return f"피부톤{int(skin_code)}"


def get_body_part_folder(pose_code: str) -> str:
    """포즈 코드 → 부위 폴더 (01-03 → 손, 04 → 발)"""
    return "손" if pose_code in ['01', '02', '03'] else "발"


def parse_session_id(session_id: str) -> Optional[Dict]:
    """
    세션 ID 파싱
    예: P131_02_02 → person_id: 131, skin_tone: 02, pose: 02
    """
    parts = session_id.strip().split('_')
    if len(parts) != 3:
        return None
    
    try:
        return {
            'person_id': int(parts[0][1:]),
            'skin_tone': parts[1],
            'pose': parts[2]
        }
    except (ValueError, IndexError):
        return None


def expand_session_to_images(session_id: str, source_root: str) -> List[str]:
    """
    세션 ID를 20개의 이미지 경로로 확장
    예: P131_02_02 → [.../피부톤2/손/P131_02_02_RGB_01.png, ..., _RGB_20.png]
    """
    info = parse_session_id(session_id)
    if not info:
        return []
    
    skin_folder = get_skin_tone_folder(info['skin_tone'])
    body_folder = get_body_part_folder(info['pose'])
    
    base_path = os.path.join(
        source_root, "1.원천데이터", "원천데이터1",
        skin_folder, body_folder
    )
    
    images = []
    for i in range(1, 21):
        filename = f"{session_id.strip()}_RGB_{i:02d}.png"
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path):
            images.append(os.path.abspath(full_path))
    
    return images


# =============================================================================
# 랜덤 분할 모드 함수
# =============================================================================
def scan_rgb_images(source_root: str) -> List[str]:
    """소스 데이터에서 RGB 이미지 경로 수집"""
    images = []
    image_base = os.path.join(source_root, "1.원천데이터", "원천데이터1")
    
    if not os.path.exists(image_base):
        print(f"오류: 이미지 폴더를 찾을 수 없습니다: {image_base}")
        return images
    
    for skin_tone in os.listdir(image_base):
        skin_path = os.path.join(image_base, skin_tone)
        if not os.path.isdir(skin_path):
            continue
            
        for body_part in ['손', '발']:
            part_path = os.path.join(skin_path, body_part)
            if not os.path.isdir(part_path):
                continue
                
            for file in os.listdir(part_path):
                if file.lower().endswith(IMAGE_EXTENSIONS) and '_RGB_' in file:
                    images.append(os.path.join(part_path, file))
    
    return images


def extract_person_ids(image_paths: List[str]) -> Set[int]:
    """이미지 경로에서 사람 ID 추출"""
    person_ids = set()
    for path in image_paths:
        filename = os.path.basename(path)
        info = extract_info_from_filename(filename)
        if info:
            person_ids.add(info['person_id'])
    return person_ids


def split_by_person(
    person_ids: Set[int], 
    ratio: Tuple[float, float, float] = DEFAULT_SPLIT_RATIO,
    seed: int = DEFAULT_SEED
) -> Tuple[List[int], List[int], List[int]]:
    """사람 ID를 train/val/test로 분할"""
    random.seed(seed)
    ids = sorted(list(person_ids))
    random.shuffle(ids)
    
    n = len(ids)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def generate_split_files(
    image_paths: List[str],
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    output_dir: str
) -> Dict[str, int]:
    """train.txt, val.txt, test.txt 생성 (랜덤 모드용)"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for path in image_paths:
        filename = os.path.basename(path)
        info = extract_info_from_filename(filename)
        if not info:
            continue
        
        person_id = info['person_id']
        abs_path = os.path.abspath(path)
        
        if person_id in train_set:
            splits['train'].append(abs_path)
        elif person_id in val_set:
            splits['val'].append(abs_path)
        elif person_id in test_set:
            splits['test'].append(abs_path)
    
    counts = {}
    for split_name, paths in splits.items():
        file_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(file_path, 'w') as f:
            f.write('\n'.join(sorted(paths)) + '\n')
        counts[split_name] = len(paths)
        print(f"  {split_name}.txt: {len(paths)}개 이미지")
    
    return counts


# =============================================================================
# 가이드 분할 모드 함수
# =============================================================================
def generate_split_files_from_guide(
    guide_dir: str,
    source_root: str,
    output_dir: str
) -> Dict[str, int]:
    """data_guide 폴더의 세션 ID 목록으로 분할 파일 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    counts = {}
    
    for split_name in ['train', 'val', 'test']:
        guide_file = os.path.join(guide_dir, f"{split_name}.txt")
        output_file = os.path.join(output_dir, f"{split_name}.txt")
        
        if not os.path.exists(guide_file):
            print(f"  경고: {guide_file} 파일을 찾을 수 없습니다.")
            counts[split_name] = 0
            continue
        
        # 세션 ID 읽기
        with open(guide_file, 'r', encoding='utf-8') as f:
            session_ids = [line.strip() for line in f if line.strip()]
        
        # 세션 ID → 이미지 경로 확장
        all_images = []
        for session_id in session_ids:
            images = expand_session_to_images(session_id, source_root)
            all_images.extend(images)
        
        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(all_images)) + '\n')
        
        counts[split_name] = len(all_images)
        print(f"  {split_name}.txt: {len(session_ids)}개 세션 → {len(all_images)}개 이미지")
    
    return counts


# =============================================================================
# 메인 함수
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='데이터 분할 스크립트')
    parser.add_argument('--source', type=str, default='datasets',
                        help='소스 데이터 루트 (default: datasets)')
    parser.add_argument('--output', type=str, default='datasets_lists',
                        help='출력 디렉토리 (default: datasets_lists)')
    
    # 상호 배타적 옵션 그룹
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--ratio', type=float, nargs=3, 
                            metavar=('TRAIN', 'VAL', 'TEST'),
                            help='랜덤 분할 비율 (예: 0.8 0.1 0.1)')
    mode_group.add_argument('--guide', type=str, 
                            help='가이드 폴더 경로 (예: data_guide)')
    
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='랜덤 시드 (default: 42, --ratio 모드에서만 사용)')
    
    args = parser.parse_args()
    
    # 기본값 설정: 둘 다 없으면 ratio 모드
    use_guide_mode = args.guide is not None
    if not use_guide_mode and args.ratio is None:
        args.ratio = list(DEFAULT_SPLIT_RATIO)
    
    print("=" * 60)
    print("데이터 분할 시작")
    print("=" * 60)
    print(f"  소스: {args.source}")
    if use_guide_mode:
        print(f"  모드: 가이드 분할")
        print(f"  가이드 폴더: {args.guide}")
    else:
        print(f"  모드: 랜덤 분할")
        print(f"  비율: {args.ratio}")
        print(f"  시드: {args.seed}")
    print()
    
    # 0. 기존 images/labels 폴더 정리
    images_dir = os.path.join(args.source, "images")
    labels_dir = os.path.join(args.source, "labels")
    
    if os.path.exists(images_dir) or os.path.exists(labels_dir):
        print("기존 images/labels 폴더 삭제 중...")
        shutil.rmtree(images_dir, ignore_errors=True)
        shutil.rmtree(labels_dir, ignore_errors=True)
        print("완료")
        print()
    
    if use_guide_mode:
        # 가이드 모드
        print("[가이드 모드] 세션 ID 기반 분할 파일 생성...")
        counts = generate_split_files_from_guide(
            guide_dir=args.guide,
            source_root=args.source,
            output_dir=args.output
        )
    else:
        # 랜덤 모드
        print("[1/3] RGB 이미지 스캔...")
        images = scan_rgb_images(args.source)
        print(f"  발견된 이미지: {len(images)}개")
        
        if not images:
            print("오류: 이미지를 찾을 수 없습니다.")
            return
        
        print("\n[2/3] ID 추출 및 분할...")
        person_ids = extract_person_ids(images)
        print(f"  발견된 인원 수: {len(person_ids)}명")
        
        train_ids, val_ids, test_ids = split_by_person(
            person_ids, 
            ratio=tuple(args.ratio), 
            seed=args.seed
        )
        print(f"  train: {len(train_ids)}명, val: {len(val_ids)}명, test: {len(test_ids)}명")
        
        print(f"\n[3/3] 분할 파일 생성 ({args.output})...")
        counts = generate_split_files(images, train_ids, val_ids, test_ids, args.output)
    
    print("\n" + "=" * 60)
    print("분할 완료")
    print(f"  총 이미지: {sum(counts.values())}개")
    print("=" * 60)


if __name__ == '__main__':
    main()
