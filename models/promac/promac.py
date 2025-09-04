
import torch
import torch.nn as nn
import clip
from PIL import Image
import numpy as np

class ProMaCModel(nn.Module):
    """Simplified ProMaC implementation"""
    def __init__(self):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32")
        self.mask_decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 224*224),
            nn.Sigmoid()
        )
    
    def forward(self, image, text):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
        
        combined = image_features + text_features
        mask = self.mask_decoder(combined.float())
        mask = mask.view(-1, 1, 224, 224)
        return mask

def load_promac():
    return ProMaCModel()

def promac_inference(image_path, prompt, model):
    """Run ProMaC inference"""
    image = Image.open(image_path).convert('RGB')
    image_input = model.preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([prompt])
    
    with torch.no_grad():
        mask = model(image_input, text_input)
    
    mask = mask.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask, mode='L')
    mask = mask.resize(image.size)
    
    return mask
