"""
Benchmark: Aspect-Pad (native batched GPU tensor) vs OpenCV (sequential CPU loop)
across a range of batch sizes, to check whether GPU kernel-launch overhead
erodes the speedup at small batch sizes.

USAGE:
    python benchmark_batch_scaling.py

NOTES:
- torch.cuda.synchronize() is called before/after every GPU-timed block.
  Without this, CUDA kernels run asynchronously and Python's wall-clock
  timer will report misleadingly fast (or misleadingly consistent) times
  that don't reflect real GPU completion time. This is the #1 mistake
  people make when benchmarking GPU code, and it's exactly what the
  kernel-launch-overhead question is probing for.
- Adjust `from aspect_pad import ...` below to match your actual v0.2.0
  API if the class/function name differs. This script assumes a
  transform that accepts a [B, C, H, W] CUDA tensor and returns a
  letterboxed [B, C, target, target] tensor.
"""

import time
import numpy as np
import cv2
import torch

# ---- Adjust this import to your actual package API ----
from aspect_pad import AspectPad  # or BatchAspectPad, etc.

TARGET_SIZE = 512
IMG_H, IMG_W = 1080, 1920
ITERATIONS = 100
BATCH_SIZES = [1, 4, 8, 16, 32, 64]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_dummy_batch(batch_size):
    """Random uint8 images, both as numpy (for cv2) and torch tensor (for Aspect-Pad)."""
    imgs_np = [
        np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]
    # [B, C, H, W] float tensor on GPU, mimicking already-loaded GPU-resident data
    batch_tensor = torch.from_numpy(np.stack(imgs_np)).permute(0, 3, 1, 2).float()
    batch_tensor = batch_tensor.to(DEVICE)
    return imgs_np, batch_tensor


def bench_opencv(imgs_np, iterations):
    """Sequential cv2 letterbox loop (CPU), no multiprocessing."""
    start = time.perf_counter()
    for _ in range(iterations):
        for img in imgs_np:
            h, w = img.shape[:2]
            scale = TARGET_SIZE / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(img, (nw, nh))
            canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
            top = (TARGET_SIZE - nh) // 2
            left = (TARGET_SIZE - nw) // 2
            canvas[top:top + nh, left:left + nw] = resized
    elapsed = time.perf_counter() - start
    return elapsed


def bench_aspect_pad(transform, batch_tensor, iterations):
    """Batched GPU tensor letterbox, properly synchronized for accurate timing."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = transform(batch_tensor)
    if DEVICE == "cuda":
        torch.cuda.synchronize()  # ensure all kernels finished before stopping the clock
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    print(f"Device: {DEVICE}")
    if DEVICE != "cuda":
        print("WARNING: no CUDA device found — GPU numbers will not be meaningful.")

    transform = AspectPad(target_size=TARGET_SIZE, fill=0)

    print(f"{'Batch':>6} | {'OpenCV (s)':>11} | {'OpenCV FPS':>11} | "
          f"{'AspectPad (s)':>14} | {'AspectPad FPS':>14} | {'Speedup':>8}")
    print("-" * 78)

    results = []
    for bs in BATCH_SIZES:
        imgs_np, batch_tensor = make_dummy_batch(bs)

        # warm-up run (important: first CUDA call includes context/cudnn init cost)
        _ = bench_aspect_pad(transform, batch_tensor, iterations=3)

        cv_time = bench_opencv(imgs_np, ITERATIONS)
        ap_time = bench_aspect_pad(transform, batch_tensor, ITERATIONS)

        total_images = bs * ITERATIONS
        cv_fps = total_images / cv_time
        ap_fps = total_images / ap_time
        speedup = cv_time / ap_time

        results.append((bs, cv_time, cv_fps, ap_time, ap_fps, speedup))
        print(f"{bs:>6} | {cv_time:>11.4f} | {cv_fps:>11.1f} | "
              f"{ap_time:>14.4f} | {ap_fps:>14.1f} | {speedup:>7.2f}x")

    print("\nRaw results (for plotting / README table):")
    print("batch_size,opencv_seconds,opencv_fps,aspectpad_seconds,aspectpad_fps,speedup")
    for row in results:
        print(",".join(str(round(v, 4)) if isinstance(v, float) else str(v) for v in row))


if __name__ == "__main__":
    main()