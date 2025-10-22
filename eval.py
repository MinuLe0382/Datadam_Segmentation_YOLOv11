import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from src.metrics import calculate_trimap_iou
from src.mask_processing import smooth_mask_gaussian, get_prediction_mask, txt_to_mask
from src.visualize import save_comparison_figure, save_low_iou_list, save_iou_distribution_figure
from src.predictor import test_time_augmentation

# --- 설정 ---
MODEL_PATH = 'outputs/runs/pretrained/weights/best.pt' # 모델 .pt 파일 경로
TEST_IMAGE_DIR = 'datasets/test/images'
GROUND_TRUTH_LABEL_DIR = 'datasets/test/labels'
SAVE_DIR = 'outputs/prediction_results'

SAVE_MASKS = True  # 예측 마스크 저장 여부
SAVE_LOW_IOU_VISUALIZATION = True # IoU가 낮은 이미지 시각화 저장 여부
LOW_IOU_THRESHOLD = 0.01

APPLY_TTA = True # Test-Time Augmentation 사용 여부
MULTI_SCALE_TTA_SCALES = [0.8, 1.0, 1.2] # TTA에 사용할 스케일

CONFIDENCE_THRESHOLD = 0.26
TRIMAP_KERNEL_SIZE = 5
# -----------

def main():
    run_name = os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH)))
    base_results_dir = os.path.join(SAVE_DIR, run_name)
    prediction_dir = os.path.join(base_results_dir, 'prediction')
    visualization_dir = os.path.join(base_results_dir, 'vis')
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    print(f"Results will be saved in: {base_results_dir}")

    model = YOLO(MODEL_PATH)

    try:
        image_files = sorted([f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f"오류: '{TEST_IMAGE_DIR}' 폴더에 이미지가 없음"); return
    except FileNotFoundError:
        print(f"오류: '{TEST_IMAGE_DIR}' 폴더를 찾을 수 없음"); return
    
    print(f"Found {len(image_files)} test images. Starting prediction...")

    total_trimap_iou, num_images_with_labels = 0, 0
    all_iou_scores = []

    for filename in tqdm(image_files, desc="Processing test images"):
        image_path = os.path.join(TEST_IMAGE_DIR, filename)
        gt_label_path = os.path.join(GROUND_TRUTH_LABEL_DIR, os.path.splitext(filename)[0] + '.txt')
        
        original_image = cv2.imread(image_path)
        if original_image is None: continue
        orig_h, orig_w = original_image.shape[:2]

        # test_time_augmentation 수행
        pred_mask_binary = test_time_augmentation(
            model, 
            original_image, 
            APPLY_TTA, 
            MULTI_SCALE_TTA_SCALES, 
            CONFIDENCE_THRESHOLD
        )
            
        final_mask = smooth_mask_gaussian(pred_mask_binary, kernel_size=(55, 55))

        if SAVE_MASKS:
            cv2.imwrite(os.path.join(prediction_dir, filename), final_mask * 255)

        gt_mask = txt_to_mask(gt_label_path, (orig_h, orig_w))
        
        if os.path.exists(gt_label_path):
            if gt_mask.sum() > 0:
                trimap_iou = calculate_trimap_iou(final_mask, gt_mask, TRIMAP_KERNEL_SIZE)
            else:
                trimap_iou = 1.0 if final_mask.sum() == 0 else 0.0

            total_trimap_iou += trimap_iou
            num_images_with_labels += 1
            all_iou_scores.append(trimap_iou)

            if trimap_iou < LOW_IOU_THRESHOLD:
                if SAVE_LOW_IOU_VISUALIZATION:
                    save_comparison_figure(
                        original_image, final_mask, gt_mask, trimap_iou,
                        os.path.join(visualization_dir, f'comparison_{filename}')
                    )
    # save_iou_distribution_figure(iou_scores: List[float], save_path: str):
    save_iou_distribution_figure(all_iou_scores, os.path.join(base_results_dir, 'iou_distribution.png'))

    if num_images_with_labels > 0:
        mean_iou = total_trimap_iou / num_images_with_labels
        print(f"\n--- Evaluation Complete ---")
        print(f"mIoU over {num_images_with_labels} images: {mean_iou:.4f}")
    else:
        print(f"정답 라벨 파일(.txt)이 없어 mIoU를 계산할 수 없습니다.")

if __name__ == '__main__':
    main()