# Aspect-Pad 📐

![CI](https://github.com/RichardHtunn/aspect-pad/actions/workflows/test.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/aspect-pad)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aspect-pad?v=2)
![PyPI - License](https://img.shields.io/pypi/l/aspect-pad?v=2)
[![Total Downloads](https://img.shields.io/pepy/dt/aspect-pad)](https://pepy.tech/project/aspect-pad)

**Letterbox resizing for PyTorch tensors, including GPU-resident batches.**

Aspect-Pad resizes and pads image tensors to a requested height and width while preserving their aspect ratio. It accepts `[C, H, W]` images and `[B, C, H, W]` batches.

The transform uses `torch.nn.functional` and runs on the input tensor's device. It uses bilinear interpolation, centers the resized image, and fills the remaining pixels with a constant value. The output retains the input dtype. Integer and half-precision tensors are interpolated in float32 before conversion back; float32 and float64 retain their precision.

## Why Aspect-Pad?
* **PyTorch only:** No OpenCV, Pillow, or NumPy runtime dependency.
* **Device aware:** Runs on CPU or CUDA according to the input tensor.
* **Batch processing:** Applies the same resize and padding to a batch of equal-size images.

---

## Performance

Aspect-Pad is intended for pipelines where tensors are already on the GPU. CPU-to-GPU transfer can outweigh any benefit if source images start on the CPU. Run `python benchmarks/benchmark.py` with `numpy` and `opencv-python` installed to compare a sequential OpenCV CPU loop with a batched Aspect-Pad tensor on your hardware.

The benchmark excludes input creation and CPU-to-GPU transfer. OpenCV receives uint8 CPU images, while Aspect-Pad receives float32 tensors on the selected device, so the measurements describe these workflows rather than an end-to-end speedup for identical inputs. CUDA timing is synchronized.

### Previous scaling measurements

NVIDIA T4 GPU and Intel Xeon CPU; 1920 × 1080 input to 512 × 512 output. These are historical measurements and will vary by hardware and input pipeline.

| Batch | OpenCV FPS | Aspect-Pad FPS | Workflow ratio |
| :--- | :--- | :--- | :--- |
| 1 | 1238.6 | 6599.0 | 5.33x |
| 4 | 1583.5 | 7068.0 | 4.59x |
| 8 | 1479.1 | 7145.7 | 4.83x |
| 16 | 1395.1 | 6969.9 | 5.00x |
| 32 | 1232.5 | 7007.6 | 5.69x |
| 64 | 1423.0 | 6845.0 | 4.81x |

---

## Installation

Install the published package from PyPI:

```bash
pip install aspect-pad
```
Python 3.7 or later is declared in the package metadata. As of 21 October 2026, PyPI lists 0.2.0 as its latest release; install `torch` separately if needed for that release. This source checkout is an unreleased `0.2.2.dev0` development snapshot and declares PyTorch as a runtime dependency.

## Usage

Aspect-Pad expects a `torch.Tensor`. The examples work on CPU and use CUDA when available.

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
# Create a dummy 1080p image tensor
device = "cuda" if torch.cuda.is_available() else "cpu"
image = torch.rand(3, 1080, 1920, device=device)

padded_image = padder(image)
print(padded_image.shape) 
# Output: torch.Size([3, 512, 512])
```

### Example 2: Batched Tensors [B, C, H, W]

```python
# Create a dummy batch of 32 1080p images
batch = torch.rand(32, 3, 1080, 1920, device=device)

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
