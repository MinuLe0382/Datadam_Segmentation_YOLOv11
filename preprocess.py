import os
import json

# 1. 원본 데이터 루트
SOURCE_ROOT_DIR = 'datasets'

# 2. 데이터셋 정보 저장 위치 (yaml 파일이 참조할 위치)
OUTPUT_TXT_DIR = os.path.abspath('datasets_lists')
os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
DATA_TYPES_TO_PROCESS = ['train', 'val', 'test']

def convert_coco_to_yolo_seg_for_folder(base_folder_path, target_class_id):
    """
    JSON을 읽어 해당 폴더 내의 'labels' 폴더에 .txt 생성
    """
    json_path = os.path.join(base_folder_path, 'annotations', 'instances_default.json')
    label_save_dir = os.path.join(base_folder_path, 'labels') # 원본 폴더 내에 labels 생성

    os.makedirs(label_save_dir, exist_ok=True)

    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        return
    except Exception as e:
        print(f"   -> 오류: {json_path} ({e})")
        return

    images_info = {img['id']: img for img in coco_data['images']}
    annotations_by_filename = {}
    
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        img_info = images_info.get(image_id)
        if not img_info: continue
        
        file_name = img_info['file_name']
        if file_name not in annotations_by_filename:
            annotations_by_filename[file_name] = []
        annotations_by_filename[file_name].append({'segmentation': ann['segmentation']})

    for img_info in coco_data.get('images', []):
        file_name = img_info['file_name']
        img_width = img_info.get('width', 1) 
        img_height = img_info.get('height', 1) 
        
        label_filename = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(label_save_dir, label_filename)
         
        #if os.path.exists(label_path): # 라벨 중복 방지
            #continue

        annotations = annotations_by_filename.get(file_name, [])
        
        with open(label_path, 'w') as f:
            if not annotations: continue
            for ann in annotations:
                class_id = target_class_id
                for seg_points in ann['segmentation']:
                    normalized_points = []
                    for i in range(0, len(seg_points), 2):
                        x = seg_points[i] / img_width
                        y = seg_points[i+1] / img_height
                        normalized_points.extend([x, y])
                    line = f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
                    f.write(line + "\n")

# 라벨 생성
def run_coco_conversion(data_type):
    print(f"\n[{data_type} 라벨 생성 (.txt)]")
    base_path = os.path.join(SOURCE_ROOT_DIR, data_type)
    if not os.path.isdir(base_path): return

    subfolders = [f.path for f in os.scandir(base_path) if f.is_dir() and f.name.startswith('CC_')]
    
    for folder in subfolders:
        try:
            if folder.endswith('04'): class_to_assign = 1
            else: class_to_assign = 0
            convert_coco_to_yolo_seg_for_folder(folder, class_to_assign)
        except Exception as e:
            print(f"Err: {folder} ({e})")

def create_image_list_file(data_type):
    """
    이미지의 절대 경로를 txt 파일에 기록.
    """
    source_base_dir = os.path.join(SOURCE_ROOT_DIR, data_type)
    
    # 저장할 텍스트 파일 경로 (예: datasets_lists/train.txt)
    list_file_path = os.path.join(OUTPUT_TXT_DIR, f'{data_type}.txt')
    
    image_paths = []
    project_root = os.getcwd()

    # 원본 폴더 순회
    for root, dirs, files in os.walk(source_base_dir):
        # images 폴더 안에 있는 파일만 대상으로 함
        if os.path.basename(root) == 'images':
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    # 절대 경로 생성
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, project_root)
                    rel_path = rel_path.replace("\\", "/")
                    image_paths.append(rel_path)
    
    with open(list_file_path, 'w') as f:
        f.write('\n'.join(image_paths))
        
    print(f" -> '{list_file_path}' 생성 완료. (이미지 수: {len(image_paths)})")

def main():
    for dtype in DATA_TYPES_TO_PROCESS:
        run_coco_conversion(dtype)   # 1. 라벨(txt) 생성
        create_image_list_file(dtype) # 2. 경로 리스트(txt) 생성

if __name__ == '__main__':
    main()