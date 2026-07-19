# InertiEAR: Automatic and Device-independent IMU-based Eavesdropping

This repository contains an implementation-oriented pipeline inspired by the paper **“InertiEAR: Automatic and Device-independent IMU-based Eavesdropping on Smartphones” (INFOCOM 2022)**.

Paper in this repo: [InertiEAR_Infocom2022.pdf](./InertiEAR_Infocom2022.pdf)

---

## 1) Project Objective (from the paper)

Based on the paper, the core objective is to show that:

- speech-related information can still be inferred from IMU data even under sub-200 Hz sampling limits,
- accelerometer + gyroscope coherence can be used for automatic segmentation,
- processing and modeling can improve **cross-device** performance for practical device-independent attacks.

The paper reports (evaluation section and comparison table):

- up to **100% segmentation success rate**,
- around **78.8% recognition accuracy**,
- up to **49.8% cross-device recognition accuracy**.

For details, see:
- [InertiEAR_Infocom2022.pdf](./InertiEAR_Infocom2022.pdf) (Abstract, Evaluation, and comparison table sections)

---

## 2) Repository File Inventory

### Top-level files/folders

| Path | Type | Purpose |
|---|---|---|
| `InertiEAR_Infocom2022.pdf` | Paper | Primary reference used for threat model, objective, segmentation idea, and evaluation claims. |
| `inertiear_pipeline/` | Source | Main Python implementation: preprocessing, features, model, dataset, training, verification. |
| `StealthyIMU_dataset/` | Dataset | Local dataset root used by training scripts (intentionally excluded from git). |
| `best_model.pth` | Artifact | Trained model weights (generated artifact; excluded). |
| `checkpoint.zip` | Artifact | Compressed checkpoint artifact (excluded). |
| `best_model.zip` | Artifact | Compressed model artifact (excluded). |
| `inertiear_kaggle.zip` | Artifact | Kaggle packaging artifact (excluded). |
| `directory_tree.txt` | Utility output | Generated file-tree dump; not required for source tracking (excluded). |
| `test_cache/` | Cache | Cached spectrogram/features for testing or preprocessing (excluded). |

### `inertiear_pipeline/` source files

| File | Purpose |
|---|---|
| `dataset.py` | Dataset loader (`InertiEARDataset`), label mapping (`CLASS_MAP`), metadata parsing, caching, segmentation-aware preprocessing flow. |
| `preprocessing.py` | Signal preprocessing primitives: Wiener filter, Butterworth filters, upsampling, Otsu thresholding, coherence-based segmentation. |
| `features.py` | Feature pipeline: dimension reduction, normalization, chronological fusion, downsampling, STFT spectrogram generation to `244x244`. |
| `model.py` | DenseNet-style model (`InertiEAR_DenseNet`), transition/layer blocks, optimizer and LR scheduler helpers. |
| `train.py` | Local training script: split dataset, train/evaluate, checkpointing, best-model saving, top-k metrics. |
| `train_kaggle.py` | Kaggle-focused training script: multi-GPU support (`DataParallel`), AMP/GradScaler, resume handling for Kaggle workflows. |
| `verify_pipeline.py` | End-to-end verification script for preprocessing, feature extraction, model forward pass, and dry-run dataset/training checks. |
| `__pycache__/` | Cache | Python bytecode cache (excluded). |

---

## 3) How this code maps to paper concepts

| Paper concept | Code implementation |
|---|---|
| IMU eavesdropping under sampling constraints | Data assumptions and training setup in `dataset.py`, `train.py`, `train_kaggle.py` |
| Coherence-based automatic segmentation | `segment_coherence()` in `preprocessing.py` |
| Noise handling (filtering) | `apply_wiener_filter()` and Butterworth filtering in `preprocessing.py` |
| Feature transformation to model-ready representation | `extract_features()` and `generate_spectrogram()` in `features.py` |
| Deep learning-based recognition | `InertiEAR_DenseNet` in `model.py`; training loops in `train.py` / `train_kaggle.py` |
| Device-independence-oriented processing pipeline | Combined preprocessing + feature normalization flow across `dataset.py`, `preprocessing.py`, and `features.py` |

---

## 4) Running the pipeline

## Requirements

Python 3 with packages used by source files:

- `numpy`
- `pandas`
- `scipy`
- `torch`
- `tqdm`
- `scikit-learn`

## Typical local training

```bash
py -3 inertiear_pipeline/train.py --csv_file StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv --data_dir StealthyIMU_dataset --cache_dir StealthyIMU_dataset/processed_cache --epochs 5 --batch_size 64 --lr 0.01
```

## Verification run

```bash
py -3 inertiear_pipeline/verify_pipeline.py
```

---

## 5) Notes on security/ethics

This repository is for research and defensive understanding of IMU side-channel privacy risks. The paper itself motivates stronger sensor-access defenses and robust countermeasures.

---

## 6) Citation

If you use this project, cite the original paper:

- Ming Gao et al., **InertiEAR: Automatic and Device-independent IMU-based Eavesdropping on Smartphones**, IEEE INFOCOM 2022.
