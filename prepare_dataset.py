import os
import json
import shutil

# 1. 원본 데이터가 있는 루트 폴더
SOURCE_ROOT_DIR = 'datasets_origin'

# 2. 최종적으로 복사할 목적지 루트 폴더
os.makedirs('datasets', exist_ok=True)
DEST_ROOT_DIR = r'datasets'

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
DATA_TYPES_TO_PROCESS = ['train', 'val', 'test']

def convert_coco_to_yolo_seg_for_folder(base_folder_path, target_class_id):
    """
    하나의 하위 폴더(예: CC_01)에 대한 COCO JSON을 
    YOLO Segmentation(.txt) 파일로 변환.
    """
    json_path = os.path.join(base_folder_path, 'annotations', 'instances_default.json')
    # [중요] 라벨 저장 위치: 원본 하위 폴더 내 'labels' 폴더
    label_save_dir = os.path.join(base_folder_path, 'labels') 

    os.makedirs(label_save_dir, exist_ok=True)

    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"   -> 경고: {json_path} 파일이 없습니다. 이 폴더를 건너뜁니다.")
        return
    except Exception as e:
        print(f"   -> 오류: {json_path} 파일 읽기 중 오류 발생. ({e})")
        return

    images_info = {img['id']: img for img in coco_data['images']}
    
    annotations_by_filename = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        img_info = images_info.get(image_id)
        if not img_info:
            continue
        
        file_name = img_info['file_name']
        if file_name not in annotations_by_filename:
            annotations_by_filename[file_name] = []
        
        annotations_by_filename[file_name].append({
            'segmentation': ann['segmentation']
        })

    for img_info in coco_data.get('images', []):
        file_name = img_info['file_name']
        img_width = img_info.get('width', 1) 
        img_height = img_info.get('height', 1) 
        if img_width == 0 or img_height == 0:
             print(f"   -> 경고: {file_name}의 너비 또는 높이가 0.")
             continue
             
        label_filename = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(label_save_dir, label_filename)
        
        annotations = annotations_by_filename.get(file_name, [])
        
        with open(label_path, 'w') as f:
            if not annotations:
                # 어노테이션이 없는 경우 빈 파일만 생성
                continue 
                
            for ann in annotations:
                class_id = target_class_id
                
                for seg_points in ann['segmentation']:
                    normalized_points = []
                    # COCO는 [x1, y1, x2, y2...] 형식이므로 2칸씩 점프
                    for i in range(0, len(seg_points), 2):
                        x = seg_points[i] / img_width
                        y = seg_points[i+1] / img_height
                        normalized_points.extend([x, y])

                    line = f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
                    f.write(line + "\n")
    
    print(f"   -> '{os.path.basename(base_folder_path)}' 폴더 라벨 생성 완료 (Class ID: {target_class_id}).")


def run_coco_conversion(data_type):
    """
    지정된 data_type(train/val/test) 폴더 내의 모든
    'CC_' 시작 하위 폴더에 대해 JSON 변환을 실행.
    """
    print(f"\n[1단계: {data_type} 데이터 변환 (JSON -> .txt)]")
    base_path = os.path.join(SOURCE_ROOT_DIR, data_type)
    
    if not os.path.isdir(base_path):
        print(f"  경고: {base_path} 폴더를 찾을 수 없음")
        return

    # 'CC_'로 시작하는 모든 하위 폴더를 찾음
    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir() and f.name.startswith('CC_')]
    
    if not subfolders:
        print(f"  '{base_path}'에서 'CC_'로 시작하는 하위 폴더를 찾지 못함")
        return
        
    print(f"  총 {len(subfolders)}개의 하위 폴더에 대해 변환개시")
    
    for folder in subfolders:
        try:
            print(f"\n  [작업 중] {folder}")
            
            # 폴더 이름 끝 두 자리를 확인하여 클래스 ID 결정
            if folder.endswith('04'):
                class_to_assign = 1
            else:
                class_to_assign = 0
            
            # 결정된 클래스 ID를 함수에 전달
            convert_coco_to_yolo_seg_for_folder(folder, class_to_assign)
            
        except Exception as e:
            print(f"   -> !!오류!!: {folder} 폴더 처리 실패 ({e})")

def collect_and_copy_files(data_type):
    """
    변환된 .txt 라벨과 원본 이미지를 최종 목적지 폴더로 복사.
    """
    print(f"\n[2단계: {data_type} 데이터 수집 및 복사]")
    source_base_dir = os.path.join(SOURCE_ROOT_DIR, data_type)
    dest_base_dir = os.path.join(DEST_ROOT_DIR, data_type)

    if not os.path.isdir(source_base_dir):
        print(f"  경고: {source_base_dir} 폴더를 찾을 수 없음")
        return

    # 최종 목적지 폴더 경로 설정
    dest_images_path = os.path.join(dest_base_dir, 'images')
    dest_labels_path = os.path.join(dest_base_dir, 'labels')

    os.makedirs(dest_images_path, exist_ok=True)
    os.makedirs(dest_labels_path, exist_ok=True)
    
    copied_images_count = 0
    copied_labels_count = 0

    print(f"  '{source_base_dir}' 폴더를 탐색하여 파일 복사를 시작합니다...")
    # 원본의 data_type 폴더(예: .../datasets_origin/val)부터 모든 하위 폴더 탐색
    for root, dirs, files in os.walk(source_base_dir):
        current_dir_name = os.path.basename(root)
        parent_dir_name = os.path.basename(os.path.dirname(root))

        if current_dir_name == 'default' and parent_dir_name == 'images':
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    source_file_path = os.path.join(root, file)
                    try:
                        shutil.copy(source_file_path, dest_images_path)
                        copied_images_count += 1
                    except shutil.Error as e:
                        print(f"  경고: {file} 이미지 복사 중 오류. {e}")

        elif current_dir_name == 'labels':
            for file in files:
                if file.lower().endswith('.txt'):
                    source_file_path = os.path.join(root, file)
                    try:
                        shutil.copy(source_file_path, dest_labels_path)
                        copied_labels_count += 1
                    except shutil.Error as e:
                        print(f"  경고: {file} 라벨 복사 중 오류. {e}")

    print(f"\n  [결과: {data_type}]")
    print(f"  총 {copied_images_count}개의 이미지 파일을 '{dest_images_path}'로 복사했습니다.")
    print(f"  총 {copied_labels_count}개의 라벨 파일을 '{dest_labels_path}'로 복사했습니다.")

def main():
    print("="*50)
    print(f"원본 위치: {SOURCE_ROOT_DIR}")
    print(f"대상 위치: {DEST_ROOT_DIR}")
    print("="*50)

    for dtype in DATA_TYPES_TO_PROCESS:
        print(f"\n{'='*20} [{dtype.upper()}] 데이터 처리 시작 {'='*20}")
        
        # COCO JSON -> YOLO .txt 변환
        run_coco_conversion(dtype)
        
        # 이미지 및 .txt 라벨 파일 수집/복사
        collect_and_copy_files(dtype)
        
        print(f"{'='*20} [{dtype.upper()}] 데이터 처리 완료 {'='*20}")

    print("\n" + "="*50)
    print("작업완료")
    print("="*50)

if __name__ == '__main__':
    main()