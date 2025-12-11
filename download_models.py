import os
import urllib.request
from pathlib import Path

def download_models():
    """Download model files during deployment"""
    model_dir = Path("COD10K Trained model")
    model_dir.mkdir(exist_ok=True)
    
    # URLs for your model files
    models = {
        "res2net50_v1b_26w_4s-3cf99910.pth": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth",
        "Net_epoch_best.pth": "https://www.dropbox.com/scl/fi/8awi5ddd72r3xz91cwwzb/Net_epoch_best.pth?rlkey=5vntbdgg5z7g8eoixfyfqm9rt&st=laauj032&dl=1"
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