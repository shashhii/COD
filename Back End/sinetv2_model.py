"""
SINet V2 Model Integration for COD Backend
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Dict, Any
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from Network_Res2Net_GRA_NCD import Network

logger = logging.getLogger(__name__)


class SINetV2Model:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.input_size = (320, 320)
        
    async def load_model(self, model_path: str = None) -> None:
        try:
            base_dir = Path(__file__).parent.parent / "COD10K Trained model"
            
            if model_path is None:
                model_path = base_dir / "Net_epoch_best.pth"
            
            pretrained_path = base_dir / "res2net50_v1b_26w_4s-3cf99910.pth"
            
                # Check if model files exist and are valid
            if not Path(model_path).exists():
                logger.warning(f"Model file not found: {model_path}")
                # Create a dummy model for deployment without trained weights
                self.model = Network(channel=32, imagenet_pretrained=False, pretrained_path=None).to(self.device)
                logger.info("Created model without pre-trained weights")
                return
            
            # Check if file is valid (not HTML)
            with open(model_path, 'rb') as f:
                header = f.read(10)
                if header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
                    logger.warning(f"Model file appears to be HTML, not a valid model: {model_path}")
                    self.model = Network(channel=32, imagenet_pretrained=False, pretrained_path=None).to(self.device)
                    logger.info("Created model without pre-trained weights due to invalid file")
                    return
            
            self.model = Network(channel=32, imagenet_pretrained=True, pretrained_path=str(pretrained_path)).to(self.device)
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            logger.info(f"SINet V2 loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load SINet V2: {e}")
            raise

    async def predict(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.model is None:
            return []
        
        try:
            input_tensor = self._preprocess_image(image)
            
            with torch.no_grad():
                S_g, S_5, S_4, S_3 = self.model(input_tensor)
                pred_mask = torch.sigmoid(S_3)
                
            detections = self._postprocess_output(pred_mask, image.shape, confidence_threshold)
            return detections
            
        except Exception as e:
            logger.error(f"SINet V2 prediction error: {e}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Normalize with ImageNet stats
        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    def _postprocess_output(self, pred_mask: torch.Tensor, original_shape: tuple, confidence_threshold: float) -> List[Dict[str, Any]]:
        detections = []
        
        try:
            mask = pred_mask.squeeze().cpu().numpy()
            h_orig, w_orig = original_shape[:2]
            mask_resized = cv2.resize(mask, (w_orig, h_orig))
            
            # Use adaptive threshold to separate objects better
            binary_mask = (mask_resized > confidence_threshold * 0.7).astype(np.uint8)
            
            # Apply morphological operations to separate connected components
            kernel = np.ones((5,5), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [x, y, x + w, y + h]
                roi_mask = mask_resized[y:y+h, x:x+w]
                confidence = float(np.mean(roi_mask))
                
                if confidence >= confidence_threshold:
                    detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'mask': self._encode_mask_roi(binary_mask, bbox),
                    })
            
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:10]
            
        except Exception as e:
            logger.error(f"Post-processing error: {e}")
            
        return detections

    def _encode_mask_roi(self, mask: np.ndarray, bbox: List[int]) -> str:
        import base64
        from io import BytesIO
        from PIL import Image
        
        try:
            x1, y1, x2, y2 = bbox
            roi_mask = mask[y1:y2, x1:x2]
            mask_img = Image.fromarray((roi_mask * 255).astype(np.uint8))
            buffer = BytesIO()
            mask_img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception:
            return ""