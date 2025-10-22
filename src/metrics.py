import cv2
import numpy as np

def calculate_trimap_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, kernel_size: int = 5) -> float:

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gt_mask_255 = (gt_mask * 255).astype(np.uint8)
    
    dilated_gt = cv2.dilate(gt_mask_255, kernel, iterations=1)
    eroded_gt = cv2.erode(gt_mask_255, kernel, iterations=1)
    
    trimap = (dilated_gt - eroded_gt) > 0
    
    evaluation_mask = ~trimap
    
    pred_eval = pred_mask[evaluation_mask]
    gt_eval = gt_mask[evaluation_mask]
    
    pred_mask_bool = pred_eval.astype(bool)
    gt_mask_bool = gt_eval.astype(bool)
    
    intersection = np.logical_and(pred_mask_bool, gt_mask_bool).sum()
    union = np.logical_or(pred_mask_bool, gt_mask_bool).sum()
    
    return float(intersection / union) if union > 0 else 0.0