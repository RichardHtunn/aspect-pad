"""CPU correctness and workflow study for the Aspect-Pad v0.2 manuscript.

Run from the repository root with NumPy, OpenCV, and PyTorch installed.
The JSON output retains every timed call and every per-case accuracy statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from aspect_pad import AspectPad


CASES = [
    (360, 640, 256, 256, 1),
    (360, 640, 256, 256, 8),
    (1080, 1920, 512, 512, 1),
    (1080, 1920, 512, 512, 4),
]
ACCURACY_CASES = [
    (96, 160, 128, 128),
    (160, 96, 128, 128),
    (31, 247, 96, 96),
    (240, 320, 192, 192),
]


def geometry(h: int, w: int, target_h: int, target_w: int) -> tuple[int, int, int, int, int, int]:
    """Use integer ratios, independently of AspectPad's floating-point scale."""
    if target_w * h <= target_h * w:
        new_w = target_w
        new_h = max(1, h * target_w // w)
    else:
        new_h = target_h
        new_w = max(1, w * target_h // h)
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    return new_h, new_w, top, target_h - new_h - top, left, target_w - new_w - left


def opencv_letterbox(images: np.ndarray, target_h: int, target_w: int) -> torch.Tensor:
    output = []
    for image in images:
        h, w = image.shape[:2]
        new_h, new_w, top, bottom, left, right = geometry(h, w, target_h, target_w)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        output.append(cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0))
    return torch.from_numpy(np.stack(output)).permute(0, 3, 1, 2).contiguous()


def direct_torch_letterbox(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Matched-tensor primitive composition, with the same dtype policy as AspectPad."""
    _, _, h, w = tensor.shape
    new_h, new_w, top, bottom, left, right = geometry(h, w, target_h, target_w)
    x = tensor if tensor.dtype in (torch.float32, torch.float64) else tensor.to(torch.float32)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return F.pad(x.to(tensor.dtype), (left, right, top, bottom), mode="constant", value=0)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def source_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def geometry_sweep(count: int, seed: int) -> dict:
    rng = random.Random(seed)
    failures = []
    legacy_failures = []
    with torch.inference_mode():
        for index in range(count):
            h, w = rng.randint(1, 251), rng.randint(1, 251)
            target_h, target_w = rng.randint(1, 191), rng.randint(1, 191)
            new_h, new_w, top, bottom, left, right = geometry(h, w, target_h, target_w)
            old_scale = min(target_w / w, target_h / h)
            old_size = (max(1, int(h * old_scale)), max(1, int(w * old_scale)))
            if old_size != (new_h, new_w):
                legacy_failures.append({"index": index, "input_hw": [h, w],
                                        "target_hw": [target_h, target_w], "old_size": old_size,
                                        "exact_size": [new_h, new_w]})
            image = torch.ones((1, h, w), dtype=torch.float32)
            output = AspectPad((target_h, target_w), fill=-17)(image)
            mask = torch.zeros((target_h, target_w), dtype=torch.bool)
            mask[top : top + new_h, left : left + new_w] = True
            content = output[0] != -17
            ok = output.shape == (1, target_h, target_w) and torch.equal(mask, content)
            if not ok:
                failures.append({"index": index, "input_hw": [h, w], "target_hw": [target_h, target_w],
                                 "expected_margins": [top, bottom, left, right]})
    return {"cases": count, "passed": count - len(failures), "failures": failures,
            "legacy_float_size_mismatches": len(legacy_failures),
            "legacy_float_examples": legacy_failures}


def pixel_comparison(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []
    with torch.inference_mode():
        for h, w, target_h, target_w in ACCURACY_CASES:
            raw = rng.integers(0, 256, (1, h, w, 3), dtype=np.uint8)
            for dtype in ("uint8", "float32"):
                images = raw if dtype == "uint8" else raw.astype(np.float32)
                tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
                actual = AspectPad((target_h, target_w))(tensor)
                reference = opencv_letterbox(images, target_h, target_w)
                difference = (actual.to(torch.float32) - reference.to(torch.float32)).abs().numpy()
                results.append({
                    "input_hw": [h, w], "target_hw": [target_h, target_w], "dtype": dtype,
                    "pixels": int(difference.size), "shape_equal": tuple(actual.shape) == tuple(reference.shape),
                    "dtype_equal": actual.dtype == reference.dtype,
                    "exact_fraction": float(np.mean(difference == 0)),
                    "mae": float(np.mean(difference)), "p99_abs_error": float(np.percentile(difference, 99)),
                    "max_abs_error": float(np.max(difference)),
                })
    return results


def measure(call, warmups: int, repeats: int) -> list[float]:
    for _ in range(warmups):
        call()
    runs = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        runs.append((time.perf_counter_ns() - start) / 1e6)
    return runs


def benchmark(blocks: int, repeats: int, warmups: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    order_rng = random.Random(seed)
    cases = []
    with torch.inference_mode():
        for h, w, target_h, target_w, batch_size in CASES:
            for dtype in ("uint8", "float32"):
                raw = rng.integers(0, 256, (batch_size, h, w, 3), dtype=np.uint8)
                images = raw if dtype == "uint8" else raw.astype(np.float32)
                tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
                transform = AspectPad((target_h, target_w))
                calls = {
                    "aspect_pad_prebuilt": lambda: transform(tensor),
                    "direct_torch_prebuilt": lambda: direct_torch_letterbox(tensor, target_h, target_w),
                    "aspect_pad_from_hwc": lambda: transform(torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()),
                    "opencv_to_bchw": lambda: opencv_letterbox(images, target_h, target_w),
                }
                if not torch.equal(calls["aspect_pad_prebuilt"](), calls["direct_torch_prebuilt"]()):
                    raise AssertionError("Matched tensor baseline differs from AspectPad")
                timings = {name: [] for name in calls}
                for _ in range(blocks):
                    names = list(calls)
                    order_rng.shuffle(names)
                    for name in names:
                        timings[name].append(measure(calls[name], warmups, repeats))
                summaries = {}
                for name, block_runs in timings.items():
                    medians = [statistics.median(runs) for runs in block_runs]
                    summaries[name] = {"block_medians_ms": medians, "median_of_block_medians_ms": statistics.median(medians),
                                       "block_median_range_ms": [min(medians), max(medians)], "runs_ms": block_runs}
                cases.append({"input_hw": [h, w], "target_hw": [target_h, target_w],
                              "batch_size": batch_size, "dtype": dtype, "timings": summaries})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research/study_v02_results.json"))
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--geometry-cases", type=int, default=120)
    args = parser.parse_args()
    if min(args.blocks, args.repeats, args.warmups, args.threads, args.geometry_cases) < 1:
        parser.error("all count arguments must be positive")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    seed = 20261021
    result = {
        "purpose": "CPU software characterization; no GPU or downstream model claim",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit(),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("aspect_pad/transform.py", "research/study_v02.py", "tests/test_transform.py")
        },
        "environment": {"cpu": cpu_model(), "python": platform.python_version(), "torch": torch.__version__,
                        "opencv": cv2.__version__, "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
                        "torch_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads()},
        "method": {"seed": seed, "blocks": args.blocks, "repeats_per_block": args.repeats,
                   "warmups_per_block": args.warmups, "timing_unit": "milliseconds per whole-batch call",
                   "prebuilt_comparison": "same preconstructed contiguous BCHW tensor; same interpolation and dtype policy",
                   "pipeline_comparison": "same preconstructed HWC NumPy arrays; both return contiguous BCHW CPU tensors; image generation excluded",
                   "output_consumption_timed": False},
        "geometry_sweep": geometry_sweep(args.geometry_cases, seed),
        "pixel_comparison": pixel_comparison(seed),
        "benchmark_cases": benchmark(args.blocks, args.repeats, args.warmups, seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"Geometry: {result['geometry_sweep']['passed']}/{args.geometry_cases} passed")
    for case in result["benchmark_cases"]:
        medians = {name: round(data["median_of_block_medians_ms"], 3) for name, data in case["timings"].items()}
        print(case["input_hw"], case["batch_size"], case["dtype"], medians)


if __name__ == "__main__":
    main()
