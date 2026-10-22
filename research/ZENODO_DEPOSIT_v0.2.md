# Proposed Zenodo deposit: Aspect-Pad preprint v0.2

Upload these two files together as one **Publication / Preprint** record after reviewing the manuscript and license choice:

1. `output/pdf/Aspect_Pad_Preprint_v0.2.pdf`
2. `output/zenodo/Aspect_Pad_Preprint_Supplement_v0.2.zip`

**Title:** Aspect-Pad: Exact Geometry and CPU Workflow Evaluation of a PyTorch Letterbox Transform

**Creator:** Shin Thant Tun

**Publication date:** Actual first public release date of this version

**Version:** 0.2

**Publisher:** Zenodo (default)

**Resource type:** Publication / Preprint

**Keywords:** image preprocessing; letterbox; aspect ratio; PyTorch; software evaluation; reproducibility

**Manuscript license:** Choose a license for the paper before publication; CC BY 4.0 is a common open choice. The bundled code remains MIT licensed as specified in `LICENSE`.

**Related identifier:** `https://github.com/RichardHtunn/aspect-pad` (software repository)

**Description:** Aspect-Pad is a PyTorch tensor letterbox transform. This software preprint specifies exact integer sizing and reports a geometry sweep, pixel comparisons with OpenCV, and CPU workflow timings. A floating-point rounding issue in the prior repository source was corrected in the supplied development source snapshot. The corrected code passed 120 sampled geometry cases and 13 unit tests. In the evaluated CPU input-to-tensor workflows, the OpenCV path had lower median latency in all eight cases. The study makes no GPU or downstream model performance claim. The supplement supplies source, tests, scripts, raw timings, and checksums.

**Suggested note:** Software preprint / technical report, v0.2. Not peer reviewed. The supplied `0.2.2.dev0` source, rather than the current PyPI release, is the evaluated implementation.

Do not combine this manuscript with the v0.1 supplement: it contains older implementation and measurement files. Do not enter a DOI in the PDF until Zenodo has reserved or assigned one. Publication is a separate step; this file is preparation metadata only.
