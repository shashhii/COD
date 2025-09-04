
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from ultralytics import YOLO

class MinimalCODSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading YOLOv8...")
        self.yolo = YOLO('yolov8n.pt')
        print("Minimal COD system ready!")
    
    def detect_objects(self, image, text_query="camouflaged object", model_type="YOLOv8", density_threshold=0.5):
        """Fallback detection using YOLOv8"""
        if image is None:
            return None, None, None
        
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            results = self.yolo(img_array, conf=density_threshold * 0.3)
            
            mask_img = Image.new('L', image.size, 0)
            bbox_img = image.copy()
            overlay_img = image.copy()
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                best_idx = np.argmax(confs)
                box = boxes[best_idx]
                class_id = int(classes[best_idx])
                class_name = self.yolo.names[class_id]
                
                x1, y1, x2, y2 = map(int, box)
                
                # Create mask
                mask_array = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
                mask_array[y1:y2, x1:x2] = 255
                mask_img = Image.fromarray(mask_array, mode='L')
                
                # Draw bbox
                draw = ImageDraw.Draw(bbox_img)
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                draw.text((x1, y1-20), f'{class_name} ({confs[best_idx]:.2f})', fill='red')
                
                # Create overlay
                if np.max(mask_array) > 0:
                    mask_colored = cv2.applyColorMap(mask_array, cv2.COLORMAP_JET)
                    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
                    blended = cv2.addWeighted(img_array, 0.7, mask_colored, 0.3, 0)
                    overlay_img = Image.fromarray(blended.astype(np.uint8))
            
            return mask_img, bbox_img, overlay_img
            
        except Exception as e:
            print(f"Detection error: {e}")
            empty_mask = Image.new('L', image.size, 0)
            return empty_mask, image, image
