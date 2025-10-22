import os
import matplotlib.pyplot as plt
import cv2
from typing import List, Dict

def save_comparison_figure(image, pred_mask, gt_mask, iou, save_path):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Prediction Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt_mask, cmap='gray')
    plt.title(f'Ground Truth Mask\nIoU: {iou:.4f}')
    plt.axis('off')
    
    plt.savefig(f"{save_path}.png")
    plt.close()

# iou가 작은 list를 확인
def save_low_iou_list(results: List[Dict], threshold: float, save_path: str):
    low_iou_items = [res for res in results if res['iou'] < threshold]
    if not low_iou_items: return

    with open(save_path, 'w') as f:
        f.write("Filename, IoU_Score\n")
        for item in sorted(low_iou_items, key=lambda x: x['iou']):
            f.write(f"{item['filename']}, {item['iou']:.4f}\n")
    print(f"Low IoU list saved to {save_path}")

# iou의 분포를 출력
def save_iou_distribution_figure(iou_scores: List[float], save_path: str):
    if not iou_scores: return

    plt.figure(figsize=(10, 6))
    plt.hist(iou_scores, bins=20, range=(0, 1), edgecolor='black')
    plt.title('IoU Score Distribution')
    plt.xlabel('IoU Score')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(save_path)
    plt.close()
    print(f"IoU distribution graph saved to {save_path}")