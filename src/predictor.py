# src/predictor.py

"""
    TTA 및 후처리를 포함한 최종 예측 마스크를 반환

    Args:
        model (YOLO): 학습된 YOLO 모델.
        original_image (np.ndarray): 원본 CV2 이미지.
        apply_tta (bool): TTA 적용 여부.
        tta_scales (List[float]): TTA에 사용할 스케일 리스트.
        conf_threshold (float): 예측 신뢰도 임계값.
        smooth_kernel_size (tuple): 가우시안 블러 커널 크기.

    Returns:
        np.ndarray: 최종 이진 마스크 (0 또는 1).
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List
from .mask_processing import get_prediction_mask

def test_time_augmentation(
    model: YOLO,
    original_image: np.ndarray,
    apply_tta: bool,
    tta_scales: List[float],
    conf_threshold: float,
) -> np.ndarray:
    
    orig_h, orig_w = original_image.shape[:2]

    if apply_tta:
        all_pred_masks = []
        images_to_test = [original_image, cv2.flip(original_image, 1)]
        
        for idx, img in enumerate(images_to_test):
            for scale in tta_scales:
                h, w = img.shape[:2]
                scaled_h, scaled_w = int(h * scale), int(w * scale)
                if scaled_h == 0 or scaled_w == 0: continue
                
                scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                
                # get_prediction_mask는 항상 (h, w) 크기의 부동 소수점 마스크를 반환
                pred_mask_scaled = get_prediction_mask(model, scaled_img, conf_threshold)
                
                # 원본 이미지 크기로 복원
                pred_mask_restored = cv2.resize(pred_mask_scaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
                if idx == 1: # 좌우 반전 이미지였다면 마스크도 다시 반전
                    pred_mask_restored = cv2.flip(pred_mask_restored, 1)
                
                all_pred_masks.append(pred_mask_restored)
        
        # 모든 TTA 예측 마스크의 평균
        final_pred_mask_float = np.mean(all_pred_masks, axis=0)

    else: # TTA 미적용
        final_pred_mask_float = get_prediction_mask(model, original_image, conf_threshold)
        
    # 이진화
    pred_mask_binary = (final_pred_mask_float > 0.5).astype(np.uint8)

    return pred_mask_binary