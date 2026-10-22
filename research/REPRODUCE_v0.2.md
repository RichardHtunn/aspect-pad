# Reproducing the Aspect-Pad v0.2 software preprint

This supplement accompanies **Aspect-Pad: Exact Geometry and CPU Workflow Evaluation of a PyTorch Letterbox Transform**, version 0.2 (21 October 2026).

## Contents

- `aspect_pad/`, `setup.py`, `README.md`, and `LICENSE`: development source snapshot `0.2.2.dev0`
- `tests/test_transform.py`: 13 functional and regression tests
- `research/study_v02.py`: geometry, pixel, and CPU workflow study
- `research/study_v02_results.json`: sampled cases, every timed call, summaries, environment metadata, and source hashes
- `research/Aspect_Pad_Preprint_v0.2.md`: editable manuscript
- `research/build_preprint_v02_pdf.py`: PDF builder and table consistency check
- `SHA256SUMS`: SHA-256 checksum for every supplied file

This development snapshot contains the integer-ratio geometry correction. The prior repository source declared `0.2.1`, while PyPI listed `0.2.0` as its latest release on 21 October 2026. Run the commands below from the extracted supplement root so Python imports the evaluated source rather than a separately installed release.

## Environment and commands

The package needs PyTorch. The study additionally needs NumPy and OpenCV; the PDF builder needs ReportLab. Use a fresh Python environment, then run:

```bash
python -m pip install -e .
python -m pip install numpy opencv-python reportlab
python -m unittest discover -s tests -v
python -m research.study_v02 --output research/reproduced_results.json
python research/build_preprint_v02_pdf.py
```

The PDF builder expects the deposited reference results at `research/study_v02_results.json` and verifies that each numerical table row agrees with that JSON. The study's `--output` option above preserves the deposited results. To check the supplied files, run `sha256sum -c SHA256SUMS` from the supplement root before generating new files. The PDF builder writes `output/pdf/Aspect_Pad_Preprint_v0.2.pdf`.

The deposited study used an AMD Ryzen 5 PRO 3500U, Python 3.13.13, PyTorch 2.13.0+cu130 with CUDA unavailable, OpenCV 5.0.0, and NumPy 2.5.1. It configured both libraries for four CPU threads. The script defaults to 120 geometry cases, three timing blocks, ten measured calls and three warm-ups per block. Changes in software, hardware, clock behavior, and system load can change timing results. No GPU or downstream model result is claimed.
