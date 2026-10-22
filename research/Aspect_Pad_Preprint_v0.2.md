# Aspect-Pad: Exact Geometry and CPU Workflow Evaluation of a PyTorch Letterbox Transform

**Software preprint / technical report, version 0.2**  
**Shin Thant Tun**  
21 October 2026

## Abstract

Letterboxing resizes an image without changing its aspect ratio and pads the result to a fixed canvas. Aspect-Pad implements this operation for channel-first PyTorch tensors without OpenCV, Pillow, or NumPy as runtime dependencies. This report specifies its exact integer geometry, tensor and dtype behavior, and measured CPU characteristics. A deterministic sweep of 120 input and target shapes exposed eight one-pixel geometry errors in the prior repository's floating-point size calculation; an integer-ratio correction passed all 120 cases and a 13-test unit suite. On four seeded synthetic images at two dtypes, differences from an OpenCV linear-resize reference were at most one intensity level for uint8 and 0.0051 on a 0-255 scale for float32. In eight CPU input-to-tensor workflows, the Aspect-Pad path took 1.9-10.6 times the median call latency of an OpenCV path on the evaluated machine. These measurements do not establish GPU behavior or downstream model quality. The contribution is a precisely specified, reproducible tensor-native software component, not a new letterboxing algorithm or a general speedup claim.

**Keywords:** image preprocessing; letterbox; aspect ratio; PyTorch; software evaluation; reproducibility

## 1. Introduction

Fixed-size image inputs are common in computer-vision pipelines. Directly stretching a rectangular image to a square canvas changes object geometry; letterboxing instead scales to fit and pads the remaining area. Aspect-Pad packages that operation for images already represented as channel-first PyTorch tensors. Its intended setting is a tensor preprocessing pipeline, including one where tensors may already reside on an accelerator. The implementation itself makes no device transfer.

This report asks three software questions: (1) does the integer geometry meet the specified canvas and centering rules across varied shapes; (2) how closely do its output pixels agree with a common OpenCV implementation; and (3) what CPU latency is observed when both a matched tensor input and a raw HWC-array-to-BCHW-tensor workflow are measured? These questions are narrower than a claim of algorithmic novelty or model-level benefit.

The principal finding is a correctness correction. The earlier implementation computed a floating-point scale and truncated products such as `w * scale`. In some cases a product mathematically equal to the target width was represented just below that integer, leaving an unintended extra padding column. The revised implementation computes sizes with integer ratios. We report the failure count for the prior formula, validation of the revised code, pixel differences, and CPU timings with raw observations in the supplement.

## 2. Related tools and scope

PyTorch supplies bilinear interpolation and constant padding as separate primitives [1, 2]. Torchvision provides configurable resize and pad transforms [3, 4]. Ultralytics provides a LetterBox transform that also handles detection labels and bounding boxes [5]. Aspect-Pad combines a fixed bilinear resize and constant padding for `[C,H,W]` or uniform `[B,C,H,W]` PyTorch tensors. It does not decode images, transform annotations, normalize channels, or accept mixed-size images within a batch.

The OpenCV path in this report is an experimental reference using `cv2.resize` with `INTER_LINEAR` and `cv2.copyMakeBorder` with constant zero fill [6]. The two libraries need not produce identical pixel values because interpolation and integer conversion details can differ. Consequently, the pixel study reports differences rather than treating exact equality with OpenCV as a correctness requirement.

## 3. Design and implementation

### 3.1 Exact resize geometry

Let the input height and width be positive integers H and W, and the target canvas be positive integers T_h and T_w. The limiting axis is chosen without floating-point division:

    if T_w * H <= T_h * W:
        W_r = T_w
        H_r = max(1, floor(H * T_w / W))
    else:
        H_r = T_h
        W_r = max(1, floor(W * T_h / H))

The resized dimensions cannot exceed the target dimensions. On the limiting axis, the resized size equals the target exactly. This follows directly from the branch assignment and does not depend on floating-point rounding. The maximum with one handles extreme aspect ratios that would otherwise truncate the other dimension to zero.

Horizontal and vertical residuals are `P_w = T_w - W_r` and `P_h = T_h - H_r`. Left and top margins are `floor(P_w/2)` and `floor(P_h/2)`; right and bottom receive the remainder. Thus opposite margins differ by at most one pixel, and the output has exactly the requested spatial shape. Aspect-Pad uses `torch.nn.functional.interpolate` with bilinear mode and `align_corners=False`, then `torch.nn.functional.pad` with constant fill [1, 2].

### 3.2 Tensor, dtype, and device semantics

The API accepts a single `[C,H,W]` tensor or a uniform `[B,C,H,W]` batch. A single image is temporarily expanded to a batch and squeezed on return. The transform rejects non-tensors, other ranks, empty spatial dimensions, empty batches or channels, and invalid target sizes. Each image in a batch shares one resize geometry.

Float32 and float64 inputs are interpolated in their original dtype. Other dtypes are converted to float32 for interpolation and converted back before padding. The output retains the original dtype, but conversion can quantize or truncate pixel values. The transform operates on the input tensor's device and does not call `.cpu()` or `.cuda()`. Accelerator execution was not measured in this study.

### 3.3 Development version and minimal example

The evaluated source snapshot is marked `0.2.2.dev0`. It corrects the repository's prior `0.2.1` source, which was not a PyPI release; PyPI listed `0.2.0` as its latest release when this report was prepared [7]. The supplement contains the exact source files used for this report. A minimal CPU example is:

    import torch
    from aspect_pad import AspectPad
    images = torch.rand(8, 3, 360, 640)
    output = AspectPad((256, 256), fill=0)(images)
    assert output.shape == (8, 3, 256, 256)

## 4. Functional and numerical validation

### 4.1 Unit tests and geometry sweep

All 13 repository unit tests passed in the evaluated environment. They cover accepted and rejected inputs, output shape, centered padding and fill, dtype retention, float64 precision, an extreme aspect ratio, and two regression shapes for the limiting-axis rounding error. These are software checks, not an assessment of downstream model accuracy.

For an independent geometry check, a seeded sweep sampled 120 positive input shapes with each dimension in 1-251 and target shapes with each dimension in 1-191. An integer-ratio reference computed the expected content rectangle. Each input was filled with one and the padding value set to -17, so the observed rectangle could be compared with the expected top, bottom, left, and right margins. The corrected implementation matched all 120 rectangles. Reapplying the old floating-point size formula to those same shapes produced eight size mismatches. This count describes the sampled sweep, not the frequency of failures in real image collections. The exact integer formula itself establishes the limiting-axis property for positive integer dimensions.

### 4.2 Pixel comparison with OpenCV

Four seeded synthetic three-channel uint8 arrays were generated with values in 0-255; float32 versions used the same values. Each implementation used the same integer target geometry and zero fill. The comparison covers 96x160 and 160x96 to 128x128, 31x247 to 96x96, and 240x320 to 192x192. Both outputs were BCHW tensors of the same dtype and shape. Table 1 reports absolute differences over the entire output, including padding. Pixel values were not normalized.

**Table 1.** Pixel difference from the OpenCV linear-resize path. MAE and maximum error are in intensity units on a 0-255 scale. Exact is the fraction of output values that agree bit for bit.

| Input to target | Dtype | Exact | MAE | Maximum error |
|---|---|---:|---:|---:|
| 96x160 to 128x128 | uint8 | 0.777 | 0.223307 | 1 |
| 96x160 to 128x128 | float32 | 0.478 | 0.000079 | 0.001534 |
| 160x96 to 128x128 | uint8 | 0.778 | 0.221863 | 1 |
| 160x96 to 128x128 | float32 | 0.476 | 0.000079 | 0.001465 |
| 31x247 to 96x96 | uint8 | 0.954 | 0.045537 | 1 |
| 31x247 to 96x96 | float32 | 0.895 | 0.000031 | 0.002396 |
| 240x320 to 192x192 | uint8 | 0.729 | 0.271367 | 1 |
| 240x320 to 192x192 | float32 | 0.344 | 0.000372 | 0.005089 |

The float32 exact-match fraction is lower than for uint8 even though its errors are much smaller than one intensity unit. These measurements characterize agreement with this reference for four synthetic images; they do not prove perceptual equivalence or compatibility with every OpenCV mode.

## 5. CPU performance study

### 5.1 Questions and timing protocol

We measured two comparisons. The *prebuilt-tensor comparison* gives both Aspect-Pad and a direct composition of the same PyTorch interpolation, conversion, and padding operations an identical preconstructed contiguous BCHW tensor. The direct composition is a matched software baseline, not an independent algorithm. The *input-to-tensor comparison* starts both paths from the same preconstructed HWC NumPy arrays and ends with contiguous BCHW CPU tensors. The Aspect-Pad path converts the array to a tensor, permutes and materializes BCHW, then calls the transform. The OpenCV path resizes and pads each HWC image, stacks the batch, and converts it to contiguous BCHW. Image generation is outside the timed region. Neither path times image decoding, output consumption, or device transfer.

The seeded cases use 360x640 inputs to a 256x256 target with batch sizes 1 and 8, and 1080x1920 inputs to a 512x512 target with batch sizes 1 and 4. Each is evaluated with uint8 and float32. Both libraries were set to four CPU threads. Each workflow had three timing blocks, each with three warm-ups and ten timed calls; workflow order was shuffled within every block. Tables 2 and 3 report the median of the three block medians in milliseconds for the whole batch. The JSON supplement contains every timed call and each block median. Timing was performed on one AMD Ryzen 5 PRO 3500U with 4 cores and 8 logical CPUs, Python 3.13.13, PyTorch 2.13.0+cu130 with CUDA unavailable, OpenCV 5.0.0, and NumPy 2.5.1.

### 5.2 Matched prebuilt tensors

**Table 2.** CPU latency with the same prebuilt contiguous BCHW input. Lower is faster. The direct PyTorch path composes the same operations and dtype policy as Aspect-Pad.

| Input to target | Batch | Dtype | Aspect-Pad (ms) | Direct PyTorch (ms) |
|---|---:|---|---:|---:|
| 360x640 to 256x256 | 1 | uint8 | 0.297 | 0.294 |
| 360x640 to 256x256 | 1 | float32 | 0.317 | 0.338 |
| 360x640 to 256x256 | 8 | uint8 | 12.076 | 11.670 |
| 360x640 to 256x256 | 8 | float32 | 6.823 | 7.240 |
| 1080x1920 to 512x512 | 1 | uint8 | 8.249 | 8.447 |
| 1080x1920 to 512x512 | 1 | float32 | 2.504 | 2.509 |
| 1080x1920 to 512x512 | 4 | uint8 | 59.104 | 59.832 |
| 1080x1920 to 512x512 | 4 | float32 | 14.981 | 12.226 |

The two paths are close in most cases. In the largest float32 batch, their three block medians ranged from 11.462 to 17.012 ms for Aspect-Pad and 11.780 to 16.787 ms for the direct composition. Those overlapping ranges caution against attributing the difference in the reported medians to wrapper overhead. No separate kernel-speed claim follows from this table.

### 5.3 HWC array to BCHW tensor

**Table 3.** CPU latency from preconstructed HWC NumPy arrays to contiguous BCHW CPU tensors. Lower is faster. Both paths receive the same values and return the same shape and dtype, but pixel values can differ as in Table 1.

| Input to target | Batch | Dtype | Aspect-Pad path (ms) | OpenCV path (ms) |
|---|---:|---|---:|---:|
| 360x640 to 256x256 | 1 | uint8 | 0.922 | 0.263 |
| 360x640 to 256x256 | 1 | float32 | 2.354 | 1.226 |
| 360x640 to 256x256 | 8 | uint8 | 14.520 | 4.432 |
| 360x640 to 256x256 | 8 | float32 | 22.662 | 11.184 |
| 1080x1920 to 512x512 | 1 | uint8 | 11.010 | 1.196 |
| 1080x1920 to 512x512 | 1 | float32 | 14.058 | 4.535 |
| 1080x1920 to 512x512 | 4 | uint8 | 80.705 | 7.619 |
| 1080x1920 to 512x512 | 4 | float32 | 76.606 | 20.581 |

The Aspect-Pad path had higher median latency in all eight measured input-to-tensor cases, by 1.9-10.6 times. The ratio is a property of these CPU workflows, not a general comparison of PyTorch and OpenCV interpolation kernels. The Aspect-Pad path materializes an input BCHW tensor before resizing, whereas the OpenCV path resizes HWC images before materializing BCHW. The uint8 path also converts to float32 for interpolation. This experiment does not isolate the cost of either step. It provides no evidence about GPU-resident inputs, host-to-device transfers, or complete inference and training loops.

## 6. Reproducibility and availability

The development source, tests, study script, raw JSON, and this manuscript are packaged in the supplement with SHA-256 checksums. From the extracted supplement root, install the package in a fresh environment and install NumPy and OpenCV for the study. Run `python -m unittest discover -s tests -v` and `python -m research.study_v02`. The script writes `research/study_v02_results.json`; use another output path to preserve the deposited file. The JSON records the seed, software versions, CPU, thread settings, source-file hashes, raw per-call timings, and accuracy results. The repository is MIT licensed; a license for the manuscript should be specified separately when depositing it.

The evaluated version is a development snapshot, not a published PyPI release. The supplement's `0.2.2.dev0` source is the authoritative implementation for this report. This preprint has not been peer reviewed.

## 7. Limitations and future work

The performance study used one laptop-class CPU and three short timing blocks per case, without locking clock frequency or temperature. It cannot support a performance ranking across machines. Its source images are synthetic seeded arrays, not decoded photographs, and the timing excludes loading and downstream consumers. The output has no annotation handling, and the correctness study does not measure bounding-box alignment, perceptual quality, or model accuracy. The 120-shape sweep is a sample, though the exact integer formula gives an analytic limiting-axis guarantee for positive integer dimensions.

The repository README includes historical T4 GPU figures based on a different workflow. They were not rerun here and are not treated as evidence for this report. Future work should evaluate matched GPU-resident and full host-to-device pipelines on multiple devices, quantify transfer and layout conversion costs separately, test real image corpora and model tasks, and measure repeated independent sessions with uncertainty intervals. Any downstream benefit claim would require such evidence. Feature work could consider configurable interpolation, antialiasing, padding modes, and annotation transforms, but these are outside the current API.

## 8. Conclusion

Aspect-Pad is a small tensor-native letterbox transform with explicitly defined geometry, dtype, and device behavior. An exact integer-ratio correction removed one-pixel sizing failures found in the prior floating-point calculation. The corrected development snapshot passed the unit suite and the sampled geometry study, and its pixel values were close to an OpenCV linear-resize reference on four synthetic images. On the measured CPU, the OpenCV input-to-tensor path was faster in every tested case. The software is therefore best presented as a convenient, reproducible tensor component whose value in GPU-resident or model-level pipelines remains to be measured.

## References

[1] PyTorch Contributors. `torch.nn.functional.interpolate` documentation. https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate

[2] PyTorch Contributors. `torch.nn.functional.pad` documentation. https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad

[3] Torchvision Contributors. `torchvision.transforms.v2.Resize` documentation. https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Resize.html

[4] Torchvision Contributors. `torchvision.transforms.v2.Pad` documentation. https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Pad.html

[5] Ultralytics. `ultralytics.data.augment.LetterBox` API reference. https://docs.ultralytics.com/reference/data/augment/

[6] OpenCV Contributors. Geometric Image Transformations (`resize`). https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html

[7] Python Package Index. `aspect-pad` project and release history. https://pypi.org/project/aspect-pad/
