# src/logger.py

"""
    평가 결과 로깅 모듈

    평가 과정에서 발생하는 메타데이터, 요약 정보, 개별 이미지의 상세 평가 결과를
    수집하고 관리하며, 최종적으로 JSON 파일로 저장.
"""

import os
import json
import datetime
from typing import List, Any, Dict

class EvaluationLogger:
    """
    평가 로그를 관리하고 JSON 파일로 저장하는 클래스
    """
    def __init__(self, model_path: str, tta_enabled: bool, conf_threshold: float):
        """
        EvaluationLogger 초기화

        Args:
            model_path (str): 평가에 사용된 모델 파일 경로.
            tta_enabled (bool): TTA(Test-Time Augmentation) 사용 여부.
            conf_threshold (float): 예측 신뢰도 임계값.
        """
        self.metadata = {
            "model_path": model_path,
            "evaluation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tta_enabled": tta_enabled,
            "conf_threshold": conf_threshold
        }
        self.details: List[Dict[str, Any]] = []
        self.total_iou = 0.0
        self.num_images = 0

    def _get_gt_classes(self, label_path: str) -> List[int]:
        """
        정답 라벨 파일(.txt)에서 고유 클래스 ID 목록을 추출

        Args:
            label_path (str): 정답 라벨 파일 경로.

        Returns:
            List[int]: 정렬된 고유 클래스 ID 리스트.
        """
        if not os.path.exists(label_path):
            return []
        with open(label_path, 'r') as f:
            # 각 줄의 첫 번째 숫자(클래스 ID)만 추출하여 중복 제거 후 정렬
            classes = {int(line.strip().split()[0]) for line in f if line.strip()}
        return sorted(list(classes))

    def update(self, filename: str, iou: float, gt_label_path: str, pred_classes: List[int]):
        """
        개별 이미지의 평가 결과를 로그에 추가

        Args:
            filename (str): 이미지 파일 이름.
            iou (float): 계산된 IoU 값.
            gt_label_path (str): 정답 라벨 파일 경로.
            pred_classes (List[int]): 모델이 예측한 클래스 ID 리스트.
        """
        self.total_iou += iou
        self.num_images += 1
        
        self.details.append({
            "filename": filename,
            "trimap_iou": round(float(iou), 4),
            "gt_classes": self._get_gt_classes(gt_label_path),
            "pred_classes": pred_classes
        })

    def save(self, output_dir: str, filename: str = 'evaluation_results.json'):
        """
        수집된 로그를 JSON 파일로 저장

        Args:
            output_dir (str): 저장할 디렉토리 경로.
            filename (str, optional): 저장할 JSON 파일 이름. Defaults to 'evaluation_results.json'.
        """
        mIoU = self.total_iou / self.num_images if self.num_images > 0 else 0.0
        
        final_log = {
            "metadata": self.metadata,
            "summary": {
                "mIoU": round(mIoU, 4),
                "total_images_evaluated": self.num_images
            },
            "details": self.details
        }

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_log, f, indent=4, ensure_ascii=False)
        print(f"Evaluation results saved to {output_path}")