import os
import urllib.request
from pathlib import Path

def download_models():
    """Download model files during deployment"""
    model_dir = Path("COD10K Trained model")
    model_dir.mkdir(exist_ok=True)
    
    # URLs for your model files (you'll need to upload these to a cloud service)
    models = {
        "res2net50_v1b_26w_4s-3cf99910.pth": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth",
        # Add your trained model URL here when you upload it
        # "Net_epoch_best.pth": "YOUR_MODEL_URL_HERE"
    }
    
    for filename, url in models.items():
        filepath = model_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_models()