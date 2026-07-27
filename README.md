# Aspect-Pad 📐

![CI](https://github.com/RichardHtunn/aspect-pad/actions/workflows/test.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/aspect-pad?v=1)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aspect-pad?v=1)
![PyPI - License](https://img.shields.io/pypi/l/aspect-pad?v=1)

A lightweight, PyTorch-compatible computer vision transform that perfectly scales and mathematically pads images without distorting their aspect ratios. 

When feeding unconstrained real-world images (like drone photography, medical scans, or OCR inputs) into Convolutional Neural Networks (CNNs), standard resizing often squishes and warps the data. **Aspect-Pad** intelligently scales the image and dynamically pads the remaining space to create a perfect square (or custom rectangle) tensor, preserving the original spatial features of your dataset.

## 🚀 Installation

You can install the package directly from PyPI:

```bash
pip install aspect-pad
```
## 💻 Quick Start

Aspect-Pad works natively with PIL Images and drops directly into standard PyTorch transforms.

```python
import torchvision.transforms as transforms
from PIL import Image
from aspect_pad import AspectPad

# Load a raw, irregularly-shaped image
raw_image = Image.open("messy_drone_photo.jpg")

# Target a 512x512 square tensor, dynamically padding the empty space with black (0)
pipeline = transforms.Compose([
    AspectPad(target_size=512, fill=0),
    transforms.ToTensor()
])

# Pass the image through the pipeline
tensor_image = pipeline(raw_image)
print(tensor_image.shape) # Output: torch.Size([3, 512, 512])
```
## ⚙️ Features

* Drop-in PyTorch Compatibility: Built to work flawlessly inside torchvision.transforms.Compose.

* Universal "Anti-Squish" Math: Automatically calculates the delta for wide or tall images and centers the original image perfectly.

* Custom Target Sizes: Target a single integer for standard square CNNs (e.g., 224 for ResNet) or pass a tuple for specific architectures (e.g., (1920, 1080)).

* Custom Fill Colors: Support for grayscale padding (e.g., 0 for black), RGB padding (e.g., (255, 255, 255) for white), or any arbitrary background color.

---
**Author:** Shin Thant Tun
