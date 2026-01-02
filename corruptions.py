import torch
import torchvision.transforms as T
from torchvision.transforms.functional import gaussian_blur
from PIL import Image



class ImageCorruptor:
    def __init__(self):
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()
    
    def blank_image_like(self,image, color=(128, 128, 128)):
        return Image.new("RGB", image.size, color)

    def light_blur(self, img):
        return gaussian_blur(img, kernel_size=5)

    def medium_blur(self, img):
        return gaussian_blur(img, kernel_size=15)

    def heavy_blur(self, img):
        return gaussian_blur(img, kernel_size=35)

    def slight_noise(self, img, std=0.2):
        x = self.to_tensor(img)
        x = torch.clamp(x + torch.randn_like(x) * std, 0, 1)
        return self.to_pil(x)
    
    def noise(self, img, std=0.5):
        x = self.to_tensor(img)
        x = torch.clamp(x + torch.randn_like(x) * std, 0, 1)
        return self.to_pil(x)

    def apply_all(self, img):
        return {
            "original": img,
            "no_visual": self.blank_image_like(img), 
            "light_blur": self.light_blur(img),
            "medium_blur": self.medium_blur(img),
            "heavy_blur": self.heavy_blur(img),
            "slight_noise": self.slight_noise(img),
            "noise": self.noise(img),
        }
