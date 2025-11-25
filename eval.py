import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from src.metrics import calculate_iou
from src.mask_processing import smooth_mask_gaussian, get_prediction_mask, txt_to_mask
from src.visualize import save_comparison_figure, save_low_iou_list, save_iou_distribution_figure
from src.predictor import test_time_augmentation
from src.logger import EvaluationLogger
import argparse

# 1. ArgumentParser 객체 생성
parser = argparse.ArgumentParser(description='YOLO Segmentation Model Evaluation Script')

# 2. 인자 추가
parser.add_argument('--model_path',
                    type=str,
                    required=True,  # 필수 인자
                    help='Path to the trained model (.pt file). (e.g., outputs/runs/train3/weights/best.pt)')

parser.add_argument('--data_list', type=str, required=True,
                    help='Path to the dataset list txt file (e.g., datasets_lists/test.txt)')

parser.add_argument('--no-save-masks',
                    action='store_false',
                    dest='save_masks',
                    help='(Flag) Do not save predicted masks. (Default: saves masks)')

parser.add_argument('--no-low-iou-vis',
                    action='store_false',
                    dest='save_low_iou_visualization',
                    help='(Flag) Do not save visualizations for low IoU images. (Default: saves visualizations)')

parser.add_argument('--low_iou_threshold',
                    type=float,
                    default=0.01,
                    help='IoU threshold below which images are saved for visualization. (Default: 0.01)')

parser.add_argument('--no-aug',
                    action='store_false',
                    dest='apply_aug',
                    help='(Flag) Do not apply Test-Time Augmentation. (Default: applies aug)')

# 3. 인자 파싱
args = parser.parse_args()

# --- 설정 ---
MODEL_PATH = args.model_path  # 모델 .pt 파일 경로 예시, 예) outputs/runs/train3/weights/best.pt
DATA_LIST_PATH = args.data_list  # 예: datasets_lists/test.txt
SAVE_DIR = 'outputs/prediction_results'

SAVE_MASKS = args.save_masks  # 예측 마스크 저장 여부
SAVE_LOW_IOU_VISUALIZATION = args.save_low_iou_visualization # IoU가 낮은 이미지 시각화 저장 여부
LOW_IOU_THRESHOLD = args.low_iou_threshold

APPLY_aug = args.apply_aug # Test-Time Augmentation 사용 여부
MULTI_SCALE_aug_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2] # aug에 사용할 스케일

CONFIDENCE_THRESHOLD = 0.26
# -----------
def get_label_path_from_image_path(img_path):
    """
    이미지 경로: .../CC_01/images/123.png
    라벨 경로: .../CC_01/labels/123.txt
    로 변환
    """
    base_name = os.path.splitext(img_path)[0] + '.txt'
    label_path = base_name.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
    return label_path

def main():
    run_name = os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH)))
    base_results_dir = os.path.join(SAVE_DIR, run_name)
    prediction_dir = os.path.join(base_results_dir, 'prediction')
    visualization_dir = os.path.join(base_results_dir, 'vis')
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"--- Evaluation Configuration ---")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Data List: {DATA_LIST_PATH}")
    print(f"Save Masks: {SAVE_MASKS}")
    print(f"Save Low IoU Vis: {SAVE_LOW_IOU_VISUALIZATION}")
    print(f"Low IoU Threshold: {LOW_IOU_THRESHOLD}")
    print(f"Apply aug: {APPLY_aug}")
    print(f"Results will be saved in: {base_results_dir}")
    print(f"---------------------------------")

    model = YOLO(MODEL_PATH)
    logger = EvaluationLogger(MODEL_PATH, APPLY_aug, CONFIDENCE_THRESHOLD)
    if not os.path.exists(DATA_LIST_PATH):
        print(f"오류: 데이터 리스트 파일을 찾을 수 없습니다: {DATA_LIST_PATH}")
        return

    with open(DATA_LIST_PATH, 'r') as f:
        # 공백 제거 및 빈 줄 제외
        image_paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(image_paths)} images in list. Starting prediction...")

    total_iou, num_images_with_labels = 0, 0
    all_iou_scores = []

    TRIMAP_KERNEL_SIZE = 11
    for image_path in tqdm(image_paths, desc="Processing test images"):
        if not os.path.exists(image_path):
            print(f"  이미지 파일 없음: {image_path}")
            continue

        filename = os.path.basename(image_path)
        gt_label_path = get_label_path_from_image_path(image_path)
        
        original_image = cv2.imread(image_path)
        if original_image is None: continue
        orig_h, orig_w = original_image.shape[:2]

        # test_time_augmentation 수행
        pred_mask_binary, pred_classes = test_time_augmentation(
            model,
            original_image,
            APPLY_aug,
            MULTI_SCALE_aug_SCALES,
            CONFIDENCE_THRESHOLD
        )
            
        final_mask = smooth_mask_gaussian(pred_mask_binary, kernel_size=(65, 65))

        if SAVE_MASKS:
            cv2.imwrite(os.path.join(prediction_dir, filename), final_mask * 255)

        gt_mask = txt_to_mask(gt_label_path, (orig_h, orig_w))
        
        if os.path.exists(gt_label_path):
            if gt_mask.sum() > 0:
                iou, intersection, union = calculate_iou(final_mask, gt_mask, TRIMAP_KERNEL_SIZE)
            else:
                iou = 1.0 if final_mask.sum() == 0 else 0.0
                intersection, union = 0, 0
                
            total_iou += iou
            num_images_with_labels += 1
            all_iou_scores.append(iou)

            logger.update(filename, iou, intersection, union, gt_label_path, pred_classes)
            if iou < LOW_IOU_THRESHOLD:
                if SAVE_LOW_IOU_VISUALIZATION:
                    save_comparison_figure(
                        original_image, final_mask, gt_mask, iou,
                        os.path.join(visualization_dir, f'comparison_{filename}')
                    )
    
    save_iou_distribution_figure(all_iou_scores, os.path.join(base_results_dir, 'iou_distribution.png'))
    
    if num_images_with_labels > 0:
        mean_iou = total_iou / num_images_with_labels
        print(f"\n--- Evaluation Complete ---")
        print(f"mIoU over {num_images_with_labels} images: {mean_iou:.4f}")
        logger.save(base_results_dir)

    else:
        print(f"정답 라벨 파일(.txt)이 없어 mIoU를 계산할 수 없습니다.")

if __name__ == '__main__':
    main()