import torch
import cv2
import sys
from pathlib import Path
import glob

sys.path.insert(0, str(Path(__file__).parent / "Back End"))
from sinetv2_model import SINetV2Model

async def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SINetV2Model(device)
    await model.load_model()
    print("Model loaded successfully\n")
    
    # Find test images
    test_paths = glob.glob(r"C:\Users\shash\Downloads\COD\Datasets\COD10K-v3\Test\Image\*.jpg")[:3]
    
    if not test_paths:
        print("No test images found!")
        return
    
    for img_path in test_paths:
        print(f"\nTesting: {Path(img_path).name}")
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        print(f"Image shape: {image.shape}")
        
        detections = await model.predict(image, confidence_threshold=0.1)
        print(f"Detections: {len(detections)}")
        if detections:
            for i, det in enumerate(detections[:2]):
                print(f"  Detection {i+1}: confidence={det['confidence']:.4f}, bbox={det['bbox']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_model())
