import time
import cv2
import numpy as np
from PIL import Image
from aspect_pad import AspectPad

def yolo_letterbox(im, new_shape=(512, 512), color=(0, 0, 0)):
    """Recreation of YOLO's official OpenCV letterbox function."""
    shape = im.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Pad using C++ memory manipulation
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def run_benchmark():
    # 1. Setup Test Data (1920x1080 Drone-style image)
    width, height = 1920, 1080
    pil_image = Image.new('RGB', (width, height), (150, 150, 150))
    
    # OpenCV uses numpy arrays formatted as (height, width, channels)
    cv2_image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2_image[:] = (150, 150, 150)

    padder = AspectPad(target_size=512)
    iterations = 1000

    print(f"🚀 Running {iterations} iterations on a {width}x{height} image...")
    print("-" * 50)

    # 2. Benchmark Aspect-Pad (PIL)
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = padder(pil_image)
    aspect_pad_time = time.perf_counter() - start_time
    print(f"Aspect-Pad (PIL):      {aspect_pad_time:.4f} seconds")

    # 3. Benchmark YOLO (OpenCV)
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = yolo_letterbox(cv2_image, new_shape=(512, 512))
    yolo_time = time.perf_counter() - start_time
    print(f"YOLO Letterbox (CV2):  {yolo_time:.4f} seconds")
    
    print("-" * 50)
    if yolo_time < aspect_pad_time:
        diff = (aspect_pad_time / yolo_time)
        print(f"Result: YOLO is {diff:.2f}x faster in raw processing.")
    else:
        diff = (yolo_time / aspect_pad_time)
        print(f"Result: Aspect-Pad is {diff:.2f}x faster in raw processing.")

if __name__ == '__main__':
    run_benchmark()