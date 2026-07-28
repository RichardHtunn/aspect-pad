# Aspect-Pad 📐

![CI](https://github.com/RichardHtunn/aspect-pad/actions/workflows/test.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/aspect-pad)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aspect-pad?v=2)
![PyPI - License](https://img.shields.io/pypi/l/aspect-pad?v=2)
[![Total Downloads](https://img.shields.io/pepy/dt/aspect-pad)](https://pepy.tech/project/aspect-pad)

**The tensor-native letterbox transform for GPU-resident pipelines.**

Aspect-Pad is a blazing-fast, PyTorch-native computer vision transform that perfectly scales and mathematically pads images without distorting their aspect ratios (commonly known as "letterboxing"). 

Unlike standard OpenCV or PIL implementations that bottleneck on the CPU, Aspect-Pad is built entirely on `torch.nn.functional`. It operates directly on native PyTorch Tensors (`[C, H, W]` or `[B, C, H, W]`), allowing you to offload padding math to the GPU and process entire batches simultaneously.

## Why Aspect-Pad?
* **100% PyTorch Native:** No OpenCV (`cv2`), `Pillow`, or `numpy` dependencies required.
* **GPU Accelerated:** Executes directly on CUDA cores.
* **Batch Processing:** Natively handles `[B, C, H, W]` dimensions to pad dozens of images simultaneously.
* **Lightweight:** A highly specific utility tool without the massive overhead of standard augmentation libraries.

---

## Performance: CPU vs. GPU

Aspect-Pad is designed specifically for environments where your data is already on the GPU (e.g., on-the-fly augmentation, batched inference, or NVIDIA DALI-style workflows). 

If you are doing CPU-side preprocessing in a standard `DataLoader`, standard OpenCV (like YOLO's `letterbox`) is highly optimized C++ and will be faster. But once your data is passed to CUDA, Aspect-Pad's native tensor implementation dominates, particularly at batch scale.

**Hardware:** Nvidia T4 GPU / Intel Xeon CPU  
**Task:** Scale & pad 1920x1080 images to 512x512 squares.

| Benchmark | YOLO (OpenCV/CPU) | Aspect-Pad (Tensor/CUDA) | Winner |
| :--- | :--- | :--- | :--- |
| **Single Image (1,000x)** | 1.04s (CPU) | 2.82s (CPU) | YOLO (2.7x Faster) |
| **Single Image (1,000x)** | 0.94s (CPU)* | **0.10s (GPU)** | **Aspect-Pad (9.0x Faster)** |
| **Batch Size 32 (100x)** | 2.37s (Looping) | **0.30s (Batched)** | **Aspect-Pad (7.8x Faster)** |

*\*OpenCV remains CPU-bound even in a CUDA environment.*

---

## Benchmark Scaling Results

| Batch | OpenCV FPS | Aspect-Pad FPS | Speedup |
| :--- | :--- | :--- | :--- |
| 1 | 1238.6 | 6599.0 | 5.33x |
| 4 | 1583.5 | 7068.0 | 4.59x |
| 8 | 1479.1 | 7145.7 | 4.83x |
| 16 | 1395.1 | 6969.9 | 5.00x |
| 32 | 1232.5 | 7007.6 | 5.69x |
| 64 | 1423.0 | 6845.0 | 4.81x |

---

## Installation

You can install the package directly from PyPI:

```bash
pip install aspect-pad
```
Requires Python >= 3.7 and PyTorch

## Usage

Aspect-Pad is designed to drop seamlessly into any modern PyTorch pipeline. It strictly expects a torch.Tensor, meaning you can pass data directly from your loader to the GPU without converting back to PIL or NumPy.

### Basic Initialization

```python
import torch
from aspect_pad import AspectPad

# Initialize the padder (creates a 512x512 square, padded with zeros/black)
padder = AspectPad(target_size=512, fill=0)

# You can also specify rectangular targets
rect_padder = AspectPad(target_size=(256, 512), fill=128)
```

### Example 1: Single Image Tensor [C, H, W]

```python
# Create a dummy 1080p image tensor and move it to GPU
image = torch.rand(3, 1080, 1920).cuda()

padded_image = padder(image)
print(padded_image.shape) 
# Output: torch.Size([3, 512, 512])
```

### Example 2: Batched Tensors [B, C, H, W]

```python
# Create a dummy batch of 32 1080p images
batch = torch.rand(32, 3, 1080, 1920).cuda()

padded_batch = padder(batch) 
print(padded_batch.shape) 
# Output: torch.Size([32, 3, 512, 512]) (Processed simultaneously)
```

### Example 3: Torchvision Compose Pipeline

```python
import torchvision.transforms.v2 as v2

# Integrate Aspect-Pad directly into a standard v2 pipeline
transform_pipeline = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    AspectPad(target_size=640, fill=0),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Author:** Shin Thant Tun
