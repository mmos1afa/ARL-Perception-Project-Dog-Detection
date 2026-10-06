# Animal Detection Using YOLO11

## ARL Perception Workshop 2026

**Final Individual Project**

Presented by the **ROAR Team**

## 1. Student and Project Identification

- **Student:** Mostafa Mohamed Atef Mohamed Ismail Kamel
- **Selected animal class:** Dog (`class_id: 0`)
- **Project type:** Individual, single-class object detection pipeline
- **Baseline architecture:** Ultralytics YOLO11n
- **Bonus architecture:** Ultralytics YOLO11s
- **Interactive notebook:** Google Colab link not supplied
- **GitHub repository:** [mmos1afa/ARL-Perception-Project-Dog-Detection](https://github.com/mmos1afa/ARL-Perception-Project-Dog-Detection)

## 2. Project Overview

This project trains and evaluates a single-class dog detector using Ultralytics YOLO11. The notebook performs the complete workflow:

1. Filter cat annotations and unpaired images from all raw datasets.
2. Collect and validate dog-only image-label pairs.
3. Create deterministic, leakage-free train, validation, and test splits.
4. Generate the YOLO dataset configuration.
5. Train YOLO11n and YOLO11s.
6. Evaluate both models on the reserved test set.
7. Benchmark latency and throughput.
8. Run inference on eight external images.
9. Display training plots, the confusion matrix, and prediction images inline.

The implementation is in [`pipeline.ipynb`](pipeline.ipynb).

## 3. Dataset Attribution and Acquisition

- **Sources:** Roboflow Universe single-class dog dataset (`yolo-dog`) and the added `dog.yolov11` dataset
- **Format:** Standard YOLO text labels:

  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```

- **Coordinate range:** Normalized to `[0.0, 1.0]`
- **License/use:** Educational and non-commercial research use

The added `dog.yolov11` source contained class `0` cat annotations and class `1` dog annotations. The filtering script ignores images containing cat annotations, skips images without labels, and remaps retained dog annotations to class `0`. The original `raw_data/` folders are not modified.

## 4. Dataset Volume and Class Distribution

- **Total usable RGB images:** 1,526
- **Total ground-truth instances:** 1,971 bounding boxes
- **Classes:** 1 (`dog`, class ID 0)
- **Mean instance density:** approximately 1.29 instances per image

Filtering summary:

- **Retained dog-only images:** 1,526
- **Skipped images containing non-dog labels:** 1
- **Skipped images without labels:** 34

The dataset includes isolated portraits, different breeds, multi-dog scenes, and images with significant background clutter.

## 5. Dataset Organization and Splitting

The notebook uses `random.Random(seed=42)` to shuffle the validated image-label pairs before copying them into the final dataset. The split is deterministic and is performed before training augmentations.

| Split | Images | Percentage | Bounding boxes | Purpose | Leakage status |
|---|---:|---:|---:|---|---|
| Train | 1,068 | 69.99% | 1,381 | Weight optimization and feature learning | Zero overlap |
| Validation | 305 | 19.99% | 386 | Hyperparameter monitoring and checkpoint selection | Zero overlap |
| Test (reserved) | 153 | 10.03% | 204 | Final unbiased benchmark | Zero overlap |
| **Total** | **1,526** | **100.00%** | **1,971** | Complete audited dataset | Verified |

The audit checks:

- No filename overlap between train, validation, and test.
- Every image has a matching label file.
- Every label row has class ID `0` and five YOLO fields.
- All normalized coordinate values are within the accepted range.

The split output is organized as:

```text
dataset/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

The filtered staging data is written to `filtered_raw_data/` by `prepare_dog_dataset.py` and can be regenerated at any time from the untouched `raw_data/` folders.

## 6. Dataset Configuration

The notebook writes `configs/data.yaml`. The portable equivalent is:

```yaml
path: ../dataset
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['dog']
```

For local execution, the notebook currently writes an absolute Windows path for `path` so Ultralytics can resolve the dataset reliably.

## 7. Model Architecture and Selection

### YOLO11n baseline

YOLO11n is the selected deployment model because the dataset contains 1,068 training images but remains moderate in scale. A larger model can memorize coat colors, backgrounds, and other dataset-specific details. The compact YOLO11n architecture provides suitable capacity while reducing overfitting risk.

The YOLO11 family includes:

- C3k2 blocks for efficient feature extraction.
- C2PSA attention for improved spatial context.
- An anchor-free detection head.
- Distribution Focal Loss for continuous bounding-box localization.
- Feature pyramid processing for targets at different scales.

### YOLO11s bonus variant

YOLO11s is trained as a comparison model. It has more parameters and higher compute cost, but the results show that the smaller YOLO11n model is better suited to this dataset and deployment target.

## 8. Training Environment and Hyperparameters

### Hardware and software

- **Computer:** Lenovo LOQ 15
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU, 6144 MiB VRAM
- **CPU:** Intel Core i5-12450HX, 8 cores / 12 threads
- **RAM:** 16 GB DDR5
- **Operating system:** Windows 11
- **Python:** 3.13.5
- **PyTorch:** 2.11.0+cu128
- **CUDA:** 12.8
- **Ultralytics:** 8.4.171
- **Torchvision:** 0.26.0+cu128

### Training settings

| Hyperparameter | YOLO11n | YOLO11s | Description |
|---|---:|---:|---|
| Input resolution | 640 x 640 | 640 x 640 | Balanced for small and large dogs |
| Batch size | 16 | 8 | YOLO11s uses 8 for 6 GB VRAM headroom |
| Epochs | 50 | 50 | Full convergence run |
| Optimizer | AdamW | AdamW | Automatic Ultralytics selection |
| Base learning rate | 0.002 | 0.002 | Selected by the optimizer configuration |
| Warmup | 3 epochs | 3 epochs | Warmup period |
| Box loss weight | 7.5 | 7.5 | Strong localization emphasis |
| Classification loss weight | 0.5 | 0.5 | Single-class classification |
| DFL loss weight | 1.5 | 1.5 | Bounding-box distribution refinement |
| Mosaic | 1.0 | 1.0 | Disabled during the final 10 epochs |
| Horizontal flip | 0.5 | 0.5 | Data augmentation |
| DataLoader workers | 0 | 0 | Prevents Windows/Jupyter multiprocessing deadlocks |
| Device | CUDA device 0 | CUDA device 0 | RTX 3050 |

The notebook uses `workers=0` because PyTorch uses process spawning on Windows. Nonzero DataLoader workers can deadlock inside an interactive Jupyter kernel.

## 9. Running the Notebook

Install the main dependency in the selected notebook kernel:

```bash
pip install ultralytics
```

For CUDA execution, PyTorch and torchvision must use compatible CUDA wheels. The verified environment uses:

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
```

Run the notebook cells in order:

1. Filter and prepare the dog-only dataset.
2. Audit the generated train/validation/test splits.
3. Train both models.
4. Evaluate the reserved test set.
5. Run latency benchmarking and comparison.
6. Run unseen-image inference.
7. Display the generated plots and predictions.

At least five images must be present in `unseen_test_images/`. This project uses eight external images.

The filtering script can also be run directly:

```bash
python prepare_dog_dataset.py --raw-dir raw_data --output-dir filtered_raw_data
```

It is safe to rerun because it recreates only `filtered_raw_data/`; it does not delete or modify `raw_data/`.

## 10. Model Checkpoints

The updated training run stores checkpoints in:

```text
runs/yolo11n/weights/best.pt
runs/yolo11s/weights/best.pt
```

The notebook's checkpoint resolver also supports the previous legacy layout:

```text
runs/detect/runs/yolo11n/weights/best.pt
runs/detect/runs/yolo11s/weights/best.pt
```

The source pretrained weights are also present at the project root:

```text
yolo11n.pt
yolo11s.pt
```

## 11. Quantitative Evaluation

Evaluation was performed strictly on the reserved test split of 153 images containing 204 ground-truth instances.

| Metric | YOLO11n baseline | YOLO11s bonus | Difference |
|---|---:|---:|---:|
| Precision | 0.8656 (86.6%) | 0.9140 (91.4%) | +4.84% YOLO11s |
| Recall | 0.8235 (82.4%) | 0.7818 (78.2%) | +4.17% YOLO11n |
| F1 score | 0.8441 (84.4%) | 0.8428 (84.3%) | +0.13% YOLO11n |
| mAP@50 | 0.8765 (87.7%) | 0.8851 (88.5%) | +0.86% YOLO11s |
| mAP@50-95 | 0.6886 (68.9%) | 0.6625 (66.3%) | +2.61% YOLO11n |

The final verified test evaluation produced approximately:

- YOLO11n: precision 0.8656, recall 0.8235, mAP@50 0.8765, mAP@50-95 0.6886
- YOLO11s: precision 0.9140, recall 0.7818, mAP@50 0.8851, mAP@50-95 0.6625

## 12. Real-World Stress Testing

The retrained YOLO11n model was tested on eight external images not used during training or validation.

| Scenario | Observation | Result | Confidence / assessment |
|---|---|---|---|
| Backlit silhouette | Dog against a sunset sky | Detected | 0.83; structural contours remained usable |
| Close-up portrait | Cropped face with no visible body | Detected | 0.91; snout and ear geometry activated features |
| Foliage occlusion | Dog partly hidden in tall grass | Detected | 0.75; localization survived foreground clutter |
| Low ambient light | Dim room and dark shadows | Detected | 0.71; detection remained stable in low contrast |
| Distant target | Dog occupied less than 2% of the frame | Detected | 0.43; small-object features remained active |
| Multi-dog pack | Three dogs interacting | Partial / grouped | 0.91 and 0.74; NMS merged two overlapping dogs |
| Extreme occlusion | Dog hidden under a couch | Missed | False negative; more than 80% occlusion and shadow |
| Hard negative | Orange kitten portrait | False positive | 0.85; single-class training generalized to a quadruped |

Prediction images are saved to:

```text
runs/unseen_eval/predictions/
```

## 13. Model Size and Efficiency Study

| Model | Parameters | GFLOPs | Checkpoint size | Latency | Throughput | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 2,582,347 | 6.4 | 5.21 MB | 14.74 ms | 67.82 FPS | 0.6886 |
| YOLO11s | 9,413,187 | 21.4 | 18.27 MB | 78.06 ms | 12.81 FPS | 0.6625 |

### Deployment verdict

YOLO11n is the recommended deployment model. With a 5.21 MB checkpoint, 6.4 GFLOPs, and approximately 68 FPS in the benchmark, it remains appropriate for real-time edge inference, robotics platforms, embedded Jetson devices, and streaming applications.

YOLO11s has approximately 3.6 times as many parameters and was substantially slower in this benchmark. It achieved higher precision and mAP@50, while YOLO11n achieved higher recall, F1, mAP@50-95, and throughput. YOLO11n remains the preferred deployment model.

## 14. Engineering Challenges and Solutions

### Windows multiprocessing deadlock

**Issue:** PyTorch DataLoader workers use process spawning on Windows. In a Jupyter kernel, nonzero worker counts can hang on inter-process communication locks.

**Solution:** Set `workers=0` and explicitly select CUDA device 0.

### Nested output paths

**Issue:** Relative project paths can be interpreted relative to Ultralytics' configured detection directory.

**Solution:** Convert the training project directory to an absolute path before calling `model.train`.

### Overlapping dogs

**Issue:** Dogs playing close together can be merged into one bounding box by non-maximum suppression.

**Potential solution:** Tune the NMS IoU threshold, for example from 0.45 toward 0.35, and review the effect on grouped scenes.

### Cross-species false positives

**Issue:** A single-class dog detector can classify visually similar quadrupeds, such as cats, as dogs.

**Potential solution:** Add 10-15% empty-background images containing non-target animals, using empty YOLO label files, and retrain.

## 15. Repository Structure

```text
.
  configs/
    data.yaml
  dataset/
    train/
    valid/
    test/
  raw_data/
  filtered_raw_data/
  runs/
    yolo11n/                    # updated YOLO11n training output
    yolo11s/                    # updated YOLO11s training output
    unseen_eval/predictions/    # annotated external-image predictions
  unseen_test_images/
  prepare_dog_dataset.py
  pipeline.ipynb
  yolo11n.pt
  yolo11s.pt
  README.md
```

## 16. License and Use

This repository is intended for educational and non-commercial research use. Dataset rights remain with the original dataset provider and should be reviewed before redistribution or commercial deployment.
