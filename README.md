# EAHN — Explanation-Aware Hybrid Network for Deepfake Detection

## Quick Start

### Local synthetic smoke test (no GPU, no data needed)
```bash
pip install -r requirements.txt
python run_full_pipeline.py \
    --dataset_name synthetic \
    --epochs 2 \
    --batch_size 2 \
    --output_dir outputs_synthetic/
```

### Kaggle Setup

**Dataset directory layout expected on disk** (Kaggle mount):
```
{data_root}/
├── manipulated_sequences/
│   ├── Deepfakes/c23/videos/*.mp4        ← label = 1 (fake)
│   ├── Face2Face/c23/videos/*.mp4        ← label = 1 (fake)
│   ├── FaceShifter/c23/videos/*.mp4      ← label = 1 (fake)
│   ├── FaceSwap/c23/videos/*.mp4         ← label = 1 (fake)
│   └── NeuralTextures/c23/videos/*.mp4   ← label = 1 (fake)
└── original_sequences/
    └── youtube/c23/videos/*.mp4          ← label = 0 (real)
```

> **No mask files exist** in this dataset version.  
> All training uses **weak supervision** (entropy + total variation loss) — `has_masks` is always `False`.

**Step 1 — Verify dataset before training:**
```bash
python scripts/verify_dataset.py \
    --data_root /kaggle/input/datasets/umardrazbhatti/ffpp-c23-custom-layout/ffpp_data
```

**Step 2 — Train (recommended 10 epochs for real results):**
```python
%cd /kaggle/working
!git clone https://github.com/umardrazbhatti/EahnCode.git
%cd EahnCode
!pip install -r requirements.txt

!python run_full_pipeline.py \
    --data_root /kaggle/input/datasets/umardrazbhatti/ffpp-c23-custom-layout/ffpp_data \
    --dataset_name ff++ \
    --dataset_compression c23 \
    --epochs 10 \
    --batch_size 4 \
    --num_workers 0 \
    --eval_after_train
```

**Expected outputs in `/kaggle/working/outputs/`:**
```
outputs/
├── best_model.pth
├── metrics.csv
├── roc_curve.png
├── pr_curve.png
├── confusion_matrix.png
├── confusion_matrix_norm.png
├── score_distribution.png
├── summary_chart.png
├── heatmaps/
│   └── {video_id}_{intrinsic,gradcam,rollout,shap}.mp4
└── explanations/
    └── {video_id}_strip.png
    └── {video_id}_explanation.txt
```

---

## Bug Fixes Applied (this version)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `config.py` | `alpha=0.2` too weak → one-hot collapse | Reverted to `alpha=0.5` (stronger entropy penalty) |
| 2 | `config.py` | `cls_dropout_p=0.3` caused train/test distribution mismatch → constant predictions | **Removed** (set to 0.0) |
| 3 | `config.py` | No backbone freezing → BN corruption with batch_size=4 | Added `freeze_backbone=True` + progressive unfreeze at epoch 3 |
| 4 | `config.py` | BCE loss ignores class imbalance | Switched default to `cls_loss_type="focal"` + `label_smoothing=0.05` |
| 5 | `config.py` | `lambda2=0.1` still too strong | Reduced to `lambda2=0.05` |
| 6 | `models/eahn.py` | `cls_dropout` branch created train/test mismatch | Removed entirely; always uses `final_feat = cls_out + attn_pool` |
| 7 | `models/eahn.py` | Attention decoupled from classification | Added `compute_gradient_saliency()` for gradient-alignment loss |
| 8 | `models/spatial_stream.py` | No way to freeze/unfreeze backbone dynamically | Added `set_frozen()` method |
| 9 | `losses/explanation.py` | Diversity computed per-frame → bypassed by augmentation noise | Computed on **per-sample time-averaged centroids** |
| 10 | `losses/explanation.py` | No class-conditional penalty → class-agnostic attention | Added `l_class_sep` hinge loss (penalise real/fake similarity > 0.2) |
| 11 | `losses/classification.py` | No label smoothing | Added `label_smoothing` parameter to BCE |
| 12 | `scripts/train_real.py` | No gradient-alignment supervision | Added `lambda_grad_align` loss every 5 batches |
| 13 | `scripts/train_real.py` | No per-class accuracy logging | Added real/fake accuracy per epoch |
| 14 | `scripts/train_real.py` | No progressive backbone unfreezing | Added unfreeze at `unfreeze_backbone_epoch` with LR reduction |
| 15 | `metrics/explanation.py` | `deletion_insertion_auc` 5D tensor indexing crash | Fixed with vectorised `torch.where()` + expanded masks |
| 16 | `utils/checkpointing.py` | `torch.load` crashes with `weights_only=True` on PyTorch 2.6+ | `weights_only=False` (safe for own checkpoints) |
| 17 | `xai/gradcam.py` | `ClassifierOutputTarget(1)` IndexError on binary output | `_ScalarOutputTarget` wrapper |

---

## Class Imbalance Handling (3 techniques combined)

1. **WeightedRandomSampler** — oversamples minority class (real) at DataLoader level.
2. **Focal Loss** — down-weights easy majority-class (fake) samples; focuses on hard examples.
3. **Label Smoothing** — prevents overconfidence on majority class; improves generalisation.
4. **Heavy Augmentation** — automatically applied to minority-class training samples when ratio > 3:1.

---

## Project Structure
```
EahnCode/
├── config.py                    # EAHNConfig dataclass + CLI override
├── requirements.txt
├── run_full_pipeline.py         # Entry point
├── README.md
├── data/
│   ├── datasets.py              # FF++, DFDC, Synthetic
│   ├── face_align.py            # MTCNN crop with tracking + disk cache
│   ├── transforms.py            # Augmentation + ImageNet normalisation
│   ├── synthetic_generator.py   # CPU-only synthetic deepfake generator
│   └── collate.py               # Custom collate for optional masks
├── models/
│   ├── eahn.py                  # EAHN: full model (fixed)
│   ├── spatial_stream.py        # EfficientNet-B4/ConvNeXt wrapper (freeze/unfreeze)
│   ├── temporal_stream.py       # 4-layer Transformer + CLS
│   └── cross_attention.py       # Cross-Attention Fusion → M_t
├── losses/
│   ├── classification.py        # BCE + Focal + label smoothing
│   ├── explanation.py           # Entropy+TV+diversity+class-separation (fixed)
│   └── temporal.py              # Gated temporal consistency
├── xai/
│   ├── gradcam.py               # Grad-CAM (binary-classifier-safe)
│   ├── attention_rollout.py     # Attention rollout over Transformer layers
│   ├── shap_explainer.py        # Integrated Gradients via Captum
│   └── sanity_checks.py         # Adebayo model-randomisation check
├── metrics/
│   ├── detection.py             # AUC-ROC, AUC-PR, F1
│   └── explanation.py           # IoU, Temporal SSIM, Faithfulness, Del/Ins AUC (fixed)
├── utils/
│   ├── checkpointing.py         # save / load (weights_only=False)
│   ├── logging_utils.py         # TensorBoard + CSV
│   └── visualization.py         # Overlay heatmaps; save MP4 + PNG strip
└── scripts/
    ├── train_synthetic.py       # Phase 1: CPU smoke test
    ├── train_real.py            # Phase 2: GPU training (fixed)
    ├── evaluate.py              # Full evaluation pipeline
    ├── dashboard.py             # Metrics table + bar chart + video display
    ├── summary_chart.py         # Two-panel summary PNG
    ├── data_analysis.py         # Dataset statistics
    └── verify_dataset.py        # Pre-training sanity check
```
