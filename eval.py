import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import argparse

from src.preprocess import prepare_data_from_list
from src.metrics import calculate_iou
from src.mask_processing import smooth_mask_gaussian, txt_to_mask
from src.visualize import save_comparison_figure, save_iou_distribution_figure
from src.predictor import test_time_augmentation
from src.logger import EvaluationLogger

# =============================================================================
parser = argparse.ArgumentParser(description='YOLO Segmentation Model Evaluation Script')

parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the trained model (.pt file)')

parser.add_argument('--data-list', type=str, default='datasets_lists/test.txt',
                    help='테스트 데이터 리스트 (default: datasets_lists/test.txt)')

parser.add_argument('--source', type=str, default='datasets',
                    help='소스 데이터 루트 (default: datasets)')

parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device (default: cuda:0)')

parser.add_argument('--no-save-masks', action='store_false', dest='save_masks',
                    help='Do not save predicted masks')

parser.add_argument('--no-low-iou-vis', action='store_false', dest='save_low_iou_visualization',
                    help='Do not save low IoU visualizations')

parser.add_argument('--low_iou_threshold', type=float, default=0.01,
                    help='IoU threshold for visualization (default: 0.01)')

parser.add_argument('--no-aug', action='store_false', dest='apply_aug',
                    help='Do not apply Test-Time Augmentation')

args = parser.parse_args()

# =============================================================================
MODEL_PATH = args.model_path
DATA_LIST_PATH = args.data_list
DEVICE = args.device
SAVE_DIR = 'outputs/prediction_results'

SAVE_MASKS = args.save_masks
SAVE_LOW_IOU_VISUALIZATION = args.save_low_iou_visualization
LOW_IOU_THRESHOLD = args.low_iou_threshold

APPLY_AUG = args.apply_aug
MULTI_SCALE_AUG_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2]
CONFIDENCE_THRESHOLD = 0.26


def get_label_path_from_image_path(img_path, labels_dir):
    """이미지 경로에서 라벨 경로 생성"""
    filename = os.path.basename(img_path)
    image_id = os.path.splitext(filename)[0]
    return os.path.join(labels_dir, f"{image_id}.txt")


def main():
    # =============================================================================
    # 데이터 준비
    # =============================================================================
    IMAGES_DIR = os.path.join(args.source, "images")
    LABELS_DIR = os.path.join(args.source, "labels")
    
    # 이미지와 라벨이 이미 존재하는지 확인
    images_exist = os.path.isdir(IMAGES_DIR) and len(os.listdir(IMAGES_DIR)) > 0
    labels_exist = os.path.isdir(LABELS_DIR) and len(os.listdir(LABELS_DIR)) > 0
    
    if images_exist and labels_exist:
        print(f"데이터가 이미 준비되어 있습니다. (images: {IMAGES_DIR}, labels: {LABELS_DIR})")
    else:
        print("데이터 준비 중...")
        success = prepare_data_from_list(
            data_list_path=DATA_LIST_PATH,
            source_root=args.source,
            images_dir=IMAGES_DIR,
            labels_dir=LABELS_DIR
        )
        if not success:
            print("데이터 준비 실패. 먼저 split.py를 실행하세요.")
            return

    # =============================================================================
    # 출력 디렉토리 설정
    # =============================================================================
    run_name = os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH)))
    if not run_name:
        run_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    
    base_results_dir = os.path.join(SAVE_DIR, run_name)
    prediction_dir = os.path.join(base_results_dir, 'prediction')
    visualization_dir = os.path.join(base_results_dir, 'vis')
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"--- Evaluation Configuration ---")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Data List: {DATA_LIST_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Apply Test-Time-Augmentation: {APPLY_AUG}")
    print(f"---------------------------------")

    # =============================================================================
    # 이미지 목록 로드
    # =============================================================================
    with open(DATA_LIST_PATH, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Found {len(image_paths)} images. Starting evaluation...")

    # =============================================================================
    # 모델 로드
    # =============================================================================
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    logger = EvaluationLogger(MODEL_PATH, APPLY_AUG, CONFIDENCE_THRESHOLD)

    # =============================================================================
    # 평가 실행
    # =============================================================================
    total_iou, num_images_with_labels = 0, 0
    all_iou_scores = []

    for image_path in tqdm(image_paths, desc="Evaluating"):
        # 심볼릭 링크 경로로 변환
        filename = os.path.basename(image_path)
        actual_image_path = os.path.join(IMAGES_DIR, filename)
        
        if not os.path.exists(actual_image_path):
            continue

        gt_label_path = get_label_path_from_image_path(image_path, LABELS_DIR)
        
        original_image = cv2.imread(actual_image_path)
        if original_image is None: 
            continue
        orig_h, orig_w = original_image.shape[:2]

        pred_mask_binary, pred_classes = test_time_augmentation(
            model, original_image, APPLY_AUG, MULTI_SCALE_AUG_SCALES, CONFIDENCE_THRESHOLD
        )
        final_mask = smooth_mask_gaussian(pred_mask_binary, kernel_size=(65, 65))

        if SAVE_MASKS:
            cv2.imwrite(os.path.join(prediction_dir, filename), final_mask * 255)

        gt_mask = txt_to_mask(gt_label_path, (orig_h, orig_w))
        
        if os.path.exists(gt_label_path):
            if gt_mask.sum() > 0:
                iou, intersection, union = calculate_iou(final_mask, gt_mask, 11)
            else:
                iou = 1.0 if final_mask.sum() == 0 else 0.0
                intersection, union = 0, 0
                
            total_iou += iou
            num_images_with_labels += 1
            all_iou_scores.append(iou)

            logger.update(filename, iou, intersection, union, gt_label_path, pred_classes)
            if iou < LOW_IOU_THRESHOLD and SAVE_LOW_IOU_VISUALIZATION:
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
        print("정답 라벨 파일이 없어 mIoU를 계산할 수 없습니다.")


if __name__ == '__main__':
    main()
