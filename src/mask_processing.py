import os
import cv2
import numpy as np
from ultralytics import YOLO

"""
    기존 마스크는 업스케일링 과정에서 가장자리 부분이 계단 현상이 발생하는데 가우시안 블러를 적용하여 이를 완화.
    그리고 다시 이진화하여 스무딩된 마스크를 생성.

    Args:
        pred_mask (np.array): 원본 이진 마스크 (range: 0 ~ 1)
        kernel_size (tuple): 가우시안 블러의 커널 사이즈, 크면 더 부드러워 짐
        threshold_val (int): 블러 처리 후 이진화할 때의 임계값.

    Returns:
        np.array: 스무딩 처리된 이진 마스크 (range: 0 ~ 1)
"""
def txt_to_mask(txt_path, image_shape):
    h, w = image_shape
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(txt_path): return gt_mask
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            poly_coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            poly_coords[:, 0] *= w
            poly_coords[:, 1] *= h
            poly_coords = poly_coords.astype(np.int32)
            cv2.fillPoly(gt_mask, [poly_coords], 1)
    return gt_mask

def get_prediction_mask(model, image, conf_threshold):
    h, w = image.shape[:2]
    results = model.predict(image, verbose=False, conf=conf_threshold)
    result = results[0]
    mask_combined = np.zeros((h, w), dtype=np.float32)
    detected_classes = []

    if result.masks is not None:
        for mask_tensor in result.masks.data:
            single_mask_float = mask_tensor.cpu().numpy()
            single_mask_resized = cv2.resize(single_mask_float, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_combined = np.maximum(mask_combined, single_mask_resized)

    if result.boxes is not None and result.boxes.cls is not None:
        detected_classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        
    unique_classes = sorted(list(set(detected_classes)))
    return mask_combined, unique_classes

def smooth_mask_gaussian(pred_mask, kernel_size=(5, 5), threshold_val=127):

    # 예측이 없는 경우
    if pred_mask.sum() == 0:
        return pred_mask

    mask_to_blur = (pred_mask * 255).astype(np.uint8)
    blurred_mask = cv2.GaussianBlur(mask_to_blur, kernel_size, 0)
    
    _, binary_mask = cv2.threshold(blurred_mask, threshold_val, 255, cv2.THRESH_BINARY)
    
    return (binary_mask // 255).astype(np.uint8)