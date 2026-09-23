from pathlib import Path

path = Path(__file__).resolve().parents[1] / "knowledge-base" / "experiment-results.md"
content = """# Project 5 Experiment Results

## Dataset coverage

- WireSeg-36K source: DeformX/WireSeg-36K, CC BY 4.0.
- WireSeg materialised sample: 300 rows across data-centre, flying-wire and wire-on-plane scenes; fixed seed 17; 209 training and 91 held-out rows.
- TUM source: Dataset for Instance Segmentation of Deformable Linear Objects; all 245 labelled images inspected across S1, S2 and S3.
- TUM adaptation split: 132 S1/S2 training images, 33 S1/S2 validation images and all 80 S3 images held out for final testing.
- MovingCables sample: inspected but excluded because the sampled flow files did not decode to a valid binary cable-mask representation.

## Measured comparison

| Method and evaluation | Precision | Recall | F1 | IoU | Mean latency | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Otsu morphology, WireSeg held-out | 0.080 | 0.146 | 0.059 | 0.033 | 3.18 ms | 314.1 |
| Weighted Tiny U-Net, WireSeg held-out | 0.079 | 0.549 | 0.115 | 0.079 | 14.19 ms | 70.5 |
| Published FASTDLO checkpoint, WireSeg held-out | 0.154 | 0.034 | 0.019 | 0.010 | 707.47 ms | 1.41 |
| Published FASTDLO checkpoint, all TUM images | 0.853 | 0.322 | 0.427 | 0.293 | 725.33 ms | 1.38 |
| TUM-fine-tuned Tiny U-Net, S3 held-out | 0.630 | 0.590 | 0.605 | 0.436 | 13.76 ms | 72.68 |
| FASTDLO ResNet-101 adapted, S3 held-out | 0.714 | 0.836 | 0.769 | 0.625 | 227.15 ms | 4.40 |

## Statistical evidence

For the adapted ResNet-101 S3 result, 2,000 fixed-seed bootstrap resamples were calculated from image-level scores. Mean F1 was 0.769 with a 95% interval of 0.763-0.775. Mean IoU was 0.625 with a 95% interval of 0.617-0.634. Mean precision was 0.714 with a 95% interval of 0.705-0.721, and mean recall was 0.836 with a 95% interval of 0.827-0.844.

## Evidence boundary

The results support an image-level segmentation comparison and a geometric grasp-candidate prototype. They do not support claims of physical grasp success, autonomous manipulation or generalisation to all real cable environments. The strongest result is a scene-held-out TUM segmentation benchmark, not an embodied robot trial. Latency figures are CPU inference measurements only and exclude image transfer, preprocessing, centreline extraction, candidate validation, calibration, robot communication and actuator motion.

Raw frame-level results are stored under `outputs/experiment/`; code-generated qualitative outputs are stored under `outputs/experiment/figures/`.
"""
path.write_text(content, encoding="utf-8")
print(path)
