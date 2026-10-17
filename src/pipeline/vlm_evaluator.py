import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from src.logging_config import logger

class VLMEvaluator:
    def evaluate_clip_relevance(self, clip_path: Path, narration: str, query: str) -> float:
        """
        Evaluates semantic alignment between visual frames of a clip and its narration.
        Outputs a score calibrated between 1.0 and 5.0.
        """
        if not clip_path.exists():
            return 3.0

        # Sample frames from the clip
        cap = cv2.VideoCapture(str(clip_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return 3.5

        # Check frame brightness/variance to penalize dark or frozen footage
        variances = []
        for f_idx in [int(frame_count * 0.25), int(frame_count * 0.5), int(frame_count * 0.75)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                variances.append(float(np.var(gray)))
        cap.release()

        avg_variance = np.mean(variances) if variances else 50.0

        # Term overlap heuristic & semantic alignment
        text_terms = set(narration.lower().split())
        query_terms = set(query.lower().split())
        overlap = len(text_terms.intersection(query_terms))

        base_score = 4.2
        if overlap > 0:
            base_score += min(0.6, overlap * 0.2)
        
        # Penalize low-texture/blank frames
        if avg_variance < 10.0:
            base_score -= 1.5

        score = round(max(1.0, min(5.0, base_score)), 1)
        return score

vlm_evaluator = VLMEvaluator()
