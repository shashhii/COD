import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.branch0 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, (1,3), padding=(0,1)),
            nn.Conv2d(out_channel, out_channel, (3,1), padding=(1,0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, (1,5), padding=(0,2)),
            nn.Conv2d(out_channel, out_channel, (5,1), padding=(2,0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, (1,7), padding=(0,3)),
            nn.Conv2d(out_channel, out_channel, (7,1), padding=(3,0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return self.relu(x_cat + self.conv_res(x))

class AggregationModule(nn.Module):
    def __init__(self, channel):
        super(AggregationModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat2 = nn.Conv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = nn.Conv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = nn.Conv2d(3*channel, channel, 3, padding=1)
    
    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = self.conv_concat2(torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1))
        x3_2 = self.conv_concat3(torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        return self.conv4(x3_2)

class SINetV2(nn.Module):
    def __init__(self, channel=32):
        super(SINetV2, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.rfb2 = RFB(512, channel)
        self.rfb3 = RFB(1024, channel)
        self.rfb4 = RFB(2048, channel)
        self.agg = AggregationModule(channel)
        self.pred = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 1)
        )
    
    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x2_rfb = self.rfb2(x2)
        x3_rfb = self.rfb3(x3)
        x4_rfb = self.rfb4(x4)
        feat = self.agg(x4_rfb, x3_rfb, x2_rfb)
        pred = self.pred(feat)
        return F.interpolate(pred, scale_factor=8, mode='bilinear', align_corners=True)

def load_model(model_path):
    model = SINetV2(channel=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def predict(model, image_path, img_size=320):
    image = Image.open(image_path).convert("RGB")
    orig_image = np.array(image)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))[0, 0].cpu()
    
    mask = (pred.numpy() * 255).astype(np.uint8)
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
    _, thresh = cv2.threshold(mask_blur, 50, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = orig_image.copy()
    scale_x = orig_image.shape[1] / img_size
    scale_y = orig_image.shape[0] / img_size
    
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output_img, 
                      (int(x*scale_x), int(y*scale_y)), 
                      (int((x+w)*scale_x), int((y+h)*scale_y)), 
                      (0, 255, 0), 2)
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(orig_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(output_img)
    plt.title("Detected Camouflaged Object")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred, output_img

if __name__ == "__main__":
    model_path = r"COD\COD10K Trained model\sinetv2_best.pth"
    model = load_model(model_path)
    print(f"✅ Model loaded from {model_path}")
    
    # Example usage
    image_path = input("Enter image path: ")
    predict(model, image_path)
