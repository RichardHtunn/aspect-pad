import time
import cv2
import numpy as np
import torch
from aspect_pad import AspectPad

def yolo_letterbox(im, new_shape=(512, 512), color=(0, 0, 0)):
    """Recreation of YOLO's official OpenCV letterbox function."""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def run_benchmark():
    width, height = 1920, 1080
    
    # 1. Setup YOLO (OpenCV/Numpy) CPU data
    cv2_image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2_image[:] = (150, 150, 150)

    # 2. Setup Aspect-Pad (PyTorch Tensor) data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tensors format is [Channels, Height, Width]
    tensor_image = torch.zeros((3, height, width), dtype=torch.float32).to(device)

    padder = AspectPad(target_size=512)
    iterations = 1000

    print(f"🚀 Running {iterations} iterations on a {width}x{height} image...")
    print(f"⚙️  PyTorch Execution Device: {device.upper()}")
    print("-" * 50)

    # Warmup PyTorch (CUDA operations are asynchronous)
    for _ in range(10):
        _ = padder(tensor_image)
    if device == "cuda":
        torch.cuda.synchronize()

    # 3. Benchmark Aspect-Pad (Native Tensor Single)
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = padder(tensor_image)
    if device == "cuda":
        torch.cuda.synchronize()
    aspect_pad_time = time.perf_counter() - start_time
    print(f"Aspect-Pad (Tensor):   {aspect_pad_time:.4f} seconds")

    # 4. Benchmark YOLO (OpenCV Single)
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = yolo_letterbox(cv2_image, new_shape=(512, 512))
    yolo_time = time.perf_counter() - start_time
    print(f"YOLO Letterbox (CV2):  {yolo_time:.4f} seconds")
    
    print("-" * 50)
    if yolo_time < aspect_pad_time:
        diff = (aspect_pad_time / yolo_time)
        print(f"Result: YOLO is {diff:.2f}x faster.")
    else:
        diff = (yolo_time / aspect_pad_time)
        print(f"Result: Aspect-Pad is {diff:.2f}x faster.")

    # 5. Batched Throughput Benchmark (Batch Size = 32)
    batch_size = 32
    print(f"\n🚀 Running Batched Benchmark (Batch Size: {batch_size}, 100 iterations)...")
    
    # Aspect-Pad batched setup
    batch_tensor = torch.zeros((batch_size, 3, height, width), dtype=torch.float32).to(device)
    
    start_time = time.perf_counter()
    for _ in range(100):
        _ = padder(batch_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    aspect_pad_batch_time = time.perf_counter() - start_time
    print(f"Aspect-Pad (Batch 32): {aspect_pad_batch_time:.4f} seconds")

    # YOLO loop setup (OpenCV cannot process 4D batches natively)
    start_time = time.perf_counter()
    for _ in range(100):
        for i in range(batch_size):
            _ = yolo_letterbox(cv2_image, new_shape=(512, 512))
    yolo_batch_time = time.perf_counter() - start_time
    print(f"YOLO (Looping 32x):    {yolo_batch_time:.4f} seconds")
    
    print("-" * 50)
    if yolo_batch_time < aspect_pad_batch_time:
        print(f"Result: YOLO is {(aspect_pad_batch_time / yolo_batch_time):.2f}x faster.")
    else:
        print(f"Result: Aspect-Pad is {(yolo_batch_time / aspect_pad_batch_time):.2f}x faster.")

if __name__ == '__main__':
    run_benchmark()