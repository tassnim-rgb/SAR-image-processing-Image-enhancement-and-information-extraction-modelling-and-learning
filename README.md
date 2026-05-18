# U-Net Pipeline: SAR & Brain MRI Segmentation
**Unit 4 Assignment  Dr. Amel Bouchemha**
*SAR Image Processing: Image Enhancement, Information Extraction, Modelling and Learning*

---

##  What This Project Does

This pipeline implements semantic image segmentation using a **U-Net architecture built from scratch in PyTorch**, applied to two imaging modalities:

- **SAR (Synthetic Aperture Radar)** : high-resolution optical and synthetic aperture radar (SAR) Earth observation imagery operated by the Korea Aerospace Research Institute (KARI)
- **Brain MRI** : segmenting tumor regions from FLAIR MRI scans

The same U-Net backbone handles both tasks. The key difference is the **preprocessing stage**, which is adapted to each modality's noise model.

---

## Notebook Structure

| Section | What It Covers |
|---------|----------------|
| `0. Setup` | Install dependencies, set random seeds, detect GPU |
| `1. Dataset Download` | Kaggle API option + synthetic data generator (works offline) |
| `2. Preprocessing` | Lee filter + log norm (SAR) · Gaussian + Z-score (MRI) |
| `3. Quality Metrics` | ENL, SSI, EPI, SSIM, PSNR with visualizations |
| `4. U-Net Architecture` | Full encoder–bottleneck–decoder with skip connections |
| `5. DataLoaders` | Dataset classes, augmentation, train/val split |
| `6. Training Loop` | BCE loss, Adam, LR scheduler, early stopping |
| `7. Training Curves` | Loss, Dice, IoU plots for both tasks |
| `8. Inference` | Visual comparison: input / ground truth / prediction |
| `9. Results Summary` | Final metrics table for both tasks |

---



## The Architecture

```
Input (1×256×256)
    │
    ▼ Encoder
  DoubleConv → 64ch  ──────────────────────────────┐ skip
  MaxPool                                           │
  DoubleConv → 128ch ─────────────────────────┐    │ skip
  MaxPool                                      │    │
  DoubleConv → 256ch ────────────────────┐     │    │ skip
  MaxPool                                │     │    │
  DoubleConv → 512ch ──────────────┐     │     │    │ skip
  MaxPool                          │     │     │    │
    │                              │     │     │    │
    ▼ Bottleneck                   │     │     │    │
  DoubleConv → 1024ch              │     │     │    │
    │                              │     │     │    │
    ▼ Decoder                      │     │     │    │
  UpConv + Concat ←────────────────┘ skip     │    │
  DoubleConv → 512ch                           │    │
  UpConv + Concat ←────────────────────────────┘    │
  DoubleConv → 256ch                                │
  UpConv + Concat ←─────────────────────────────────┘
  DoubleConv → 128ch
  UpConv + Concat ← skip (64ch)
  DoubleConv → 64ch
    │
    ▼ Output
  Conv 1×1 → 1ch (logits)
  Sigmoid → probability map
```
---

## References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- Lee, *Digital Image Enhancement and Noise Filtering by Use of Local Statistics*, IEEE TPAMI 1980
- Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP 2004
- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
- Buda et al., *Association of genomic subtypes of lower-grade gliomas with shape features*, CBM 2019

---

*Assignment for Unit 4 SAR Image Processing: Image Enhancement and Information Extraction, Modelling and Learning*
*Supervisor: Dr. Amel Bouchemha*
