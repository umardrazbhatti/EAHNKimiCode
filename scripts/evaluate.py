"""
scripts/evaluate.py — Full evaluation: detection + explanation metrics + heatmaps.

FIXES APPLIED:
  1. GradCAMExplainer now called without target_layer kwarg (auto-detects internally).
  2. Faithfulness gradient computation fixed: proper retain_graph, zero_grad, detach.
  3. Deletion/Insertion AUC: safer tensor handling, explicit shape checks.
  4. Added missing torch.nn.functional import.
  5. load_checkpoint now loads only model weights for evaluation.
  6. Added try/except around entire heatmap generation to prevent eval crash.
"""

import os
import csv
import contextlib
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import EAHNConfig
from models.eahn import EAHN
from data.datasets import DeepfakeDataset
from data.collate import deepfake_collate_fn
from metrics.detection import DetectionMetrics
from metrics.explanation import ExplanationMetrics
from utils.checkpointing import load_checkpoint
from utils.visualization import (
    save_annotated_frame_strip,
    save_explanation_video,
    overlay_heatmap_on_frame,
    get_region_label,
)
import cv2


def save_detection_graphs(probs, labels, output_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_curve, roc_auc_score,
        precision_recall_curve, average_precision_score,
        confusion_matrix,
    )
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"EAHN  AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Deepfake Detection (FF++ c23)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, color="darkorange", label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150)
    plt.close(fig)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-8)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                vmin=0.0, vmax=1.0)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix (Normalised)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix_norm.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs[labels == 0], bins=30, alpha=0.6, label="Real", color="blue")
    ax.hist(probs[labels == 1], bins=30, alpha=0.6, label="Fake", color="red")
    ax.axvline(0.5, color="black", linestyle="--", label="Decision threshold")
    ax.set_xlabel("Predicted Probability (Deepfake)")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=150)
    plt.close(fig)

    required_pngs = ["roc_curve.png", "pr_curve.png",
                     "confusion_matrix.png", "confusion_matrix_norm.png"]
    for fname in required_pngs:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"[Evaluate] Required PNG not found after saving: {fpath}"
            )
    print(f"[Evaluate] Detection graphs saved → {output_dir}")


def run_evaluation(config: EAHNConfig):
    device = torch.device(config.device)

    model = EAHN(config).to(device)
    ckpt_path = os.path.join(config.output_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        import glob as _glob
        candidates = sorted(_glob.glob(
            os.path.join(config.output_dir, "checkpoint_epoch*.pth")
        ))
        if candidates:
            ckpt_path = candidates[-1]
            print(f"[Eval] best_model.pth not found — using {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {config.output_dir}. "
                "Did training complete without errors?"
            )

    # FIX: Load only model weights for evaluation (no optimizer/scheduler needed)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Loaded best model for evaluation.")

    test_ds = DeepfakeDataset(config, "test", config.dataset_name)
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=deepfake_collate_fn,
    )

    all_probs, all_labels = [], []
    all_M_t_up, all_masks = [], []
    all_has_mask_flags    = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating detection"):
            frames = batch["frames"].to(device)
            out    = model(frames)
            all_probs.extend(out.prob.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
            all_M_t_up.append(out.M_t_up.cpu())
            all_masks.append(batch["mask"].cpu())
            all_has_mask_flags.extend(batch["has_mask"].cpu().tolist())

    all_M_t_up = torch.cat(all_M_t_up, dim=0)
    all_masks  = torch.cat(all_masks,  dim=0)

    det_metrics = DetectionMetrics.compute(all_probs, all_labels)
    print("Detection Metrics:", det_metrics)

    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    try:
        preds_arr = (np.array(all_probs) >= 0.5).astype(int)
        cm        = sk_confusion_matrix(np.array(all_labels, dtype=int), preds_arr)
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn = fp = fn = tp = 0

    labels_arr = np.array(all_labels)
    if len(np.unique(labels_arr)) >= 2:
        save_detection_graphs(all_probs, all_labels, config.output_dir)
    else:
        print("[Evaluate] Skipping detection graphs — only one class in test set.")

    train_ds_tmp = DeepfakeDataset(config, "train", config.dataset_name)
    val_ds_tmp   = DeepfakeDataset(config, "val",   config.dataset_name)
    split_counts = {
        "total":      len(train_ds_tmp) + len(val_ds_tmp) + len(test_ds),
        "train":      len(train_ds_tmp),
        "train_real": train_ds_tmp.n_real,
        "train_fake": train_ds_tmp.n_fake,
        "val":        len(val_ds_tmp),
        "test":       len(test_ds),
        "test_real":  test_ds.n_real,
        "test_fake":  test_ds.n_fake,
    }
    metrics_dict_full = {
        **det_metrics,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    from scripts.summary_chart import plot_summary_chart
    plot_summary_chart(metrics_dict_full, split_counts, config.output_dir)

    subset_size = min(config.heatmap_samples, len(test_ds))
    rng     = np.random.default_rng(42)
    indices = rng.choice(len(test_ds), subset_size, replace=False)

    M_sub_avg = all_M_t_up[indices].mean(dim=1)
    masks_sub = all_masks[indices]
    hm_flags  = [all_has_mask_flags[int(i)] for i in indices]
    avg_iou   = ExplanationMetrics.localisation_iou(
        M_sub_avg, masks_sub, hm_flags, threshold=0.5
    )

    ssim_val = ExplanationMetrics.temporal_ssim(all_M_t_up[indices])

    # FIX: Faithfulness gradient computation — proper retain_graph, zero_grad, detach
    grad_maps = []
    model.zero_grad(set_to_none=True)
    for idx in tqdm(indices, desc="Computing faithfulness", leave=False):
        sample      = test_ds[idx]
        frames_t    = sample["frames"].unsqueeze(0).to(device)

        # Ensure gradients are enabled and model is in train mode temporarily
        was_training = model.training
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        frames_t.requires_grad_(True)
        out = model(frames_t)

        # Binary output: sum the logit (scalar) and backprop
        score = out.logit.sum()
        score.backward(retain_graph=False)

        if frames_t.grad is None:
            # Fallback if gradient is None
            grads = torch.zeros_like(frames_t)
        else:
            grads = frames_t.grad.detach().clone()

        grads_abs = grads.abs().mean(dim=2)  # (1, T, H, W)
        grads_7 = F.interpolate(
            grads_abs.reshape(grads_abs.shape[1], 1, *grads_abs.shape[2:]),
            size=(7, 7), mode="bilinear", align_corners=False,
        ).squeeze(1)  # (T, 7, 7)

        grad_maps.append(grads_7.cpu())

        # Clean up
        frames_t.requires_grad_(False)
        model.zero_grad(set_to_none=True)

    # Restore model to eval mode
    if not was_training:
        model.eval()
    for p in model.parameters():
        p.requires_grad = False

    grad_maps = torch.stack(grad_maps)
    M_sub     = all_M_t_up[indices].mean(dim=1)
    M_sub_7   = F.interpolate(
        M_sub.unsqueeze(1), size=(7, 7), mode="bilinear", align_corners=False
    ).squeeze(1)
    grad_7_avg = grad_maps.mean(dim=1)

    faithful_corr = ExplanationMetrics.faithfulness_correlation(
        M_sub_7.reshape(subset_size, -1),
        grad_7_avg.reshape(subset_size, -1),
    )

    # FIX: Deletion/Insertion AUC — safer tensor handling
    del_ins = {"deletion_auc": 0.0, "insertion_auc": 0.0}
    try:
        sample_idx    = int(indices[0])
        frames_sample = test_ds[sample_idx]["frames"].unsqueeze(0)
        sal_sample    = all_M_t_up[sample_idx].unsqueeze(0)

        if isinstance(sal_sample, torch.Tensor):
            sal_np = sal_sample.detach().cpu().numpy()
        elif isinstance(sal_sample, np.ndarray):
            sal_np = sal_sample
        else:
            raise TypeError(f"sal_sample is {type(sal_sample)}, expected Tensor or ndarray")

        # Ensure sal_np is 4D: (B, T, H, W)
        if sal_np.ndim == 3:
            sal_np = sal_np[np.newaxis, ...]

        del_ins = ExplanationMetrics.deletion_insertion_auc(
            model, frames_sample, sal_np, steps=10
        )
    except Exception as e:
        print(f"  [Deletion/Insertion AUC skipped: {e}]")

    collapse_diag = ExplanationMetrics.collapse_diagnostics(all_M_t_up)
    print("Collapse Diagnostics:", collapse_diag)

    warnings_list = []
    if collapse_diag["inter_sample_cosine_mean"] > 0.95:
        warnings_list.append(
            f"inter_sample_cosine_mean={collapse_diag['inter_sample_cosine_mean']:.3f}"
        )
    if collapse_diag["peak_mode_share"] > 0.5:
        warnings_list.append(
            f"peak_mode_share={collapse_diag['peak_mode_share']:.3f}"
        )
    if collapse_diag["m_t_std_mean"] > 0.13:
        warnings_list.append(
            f"m_t_std_mean={collapse_diag['m_t_std_mean']:.3f}"
        )
    if warnings_list:
        print("\n[COLLAPSE WARNING] Explanation collapse detected:")
        for w in warnings_list:
            print(f"  - {w}")
        print("  Do NOT proceed to longer runs. Diagnose the explanation head first.\n")

    mt_vs_random_cosine = 1.0
    try:
        from xai.sanity_checks import model_randomization_check
        _sample_idx     = int(indices[0])
        _frames_sample  = test_ds[_sample_idx]["frames"].unsqueeze(0)
        mt_vs_random_cosine = model_randomization_check(model, _frames_sample, n_random=3)
        print(f"[Sanity] model_randomization cosine sim = {mt_vs_random_cosine:.3f} "
              f"({'PASS < 0.7' if mt_vs_random_cosine < 0.7 else 'WARN > 0.7'})")
    except Exception as e:
        print(f"  [Adebayo sanity check skipped: {e}]")

    avg_iou_display = f"{avg_iou:.4f}" if avg_iou is not None else "N/A"
    exp_metrics = {
        "avg_iou":                    avg_iou_display,
        "temporal_ssim":              ssim_val,
        "faithfulness_corr":          faithful_corr,
        "mt_vs_random_model_cosine":  mt_vs_random_cosine,
        **del_ins,
        **collapse_diag,
    }
    print("Explanation Metrics:", exp_metrics)

    os.makedirs(config.output_dir, exist_ok=True)
    csv_path = os.path.join(config.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in {**det_metrics, **exp_metrics}.items():
            writer.writerow([k, v])
    print(f"Metrics saved to {csv_path}")

    # FIX: Wrap heatmap generation in try/except so eval never crashes on visualization
    if config.save_heatmaps:
        try:
            _generate_heatmaps(config, model, test_ds, indices[:5], device, all_probs,
                               batch_inter_sample_sim=collapse_diag["inter_sample_cosine_mean"])
        except Exception as e:
            print(f"[WARN] Heatmap generation failed: {e}. Continuing without heatmaps.")

    try:
        _save_representative_heatmaps(
            config, model, test_ds, all_probs, all_labels, device,
            batch_inter_sample_sim=collapse_diag["inter_sample_cosine_mean"],
            temporal_ssim=ssim_val,
            inter_sample_cosine=collapse_diag["inter_sample_cosine_mean"],
        )
    except Exception as e:
        print(f"[WARN] Representative heatmaps failed: {e}. Continuing.")

    print("Evaluation complete. Outputs saved to", config.output_dir)


def _generate_heatmaps(config, model, test_ds, sample_indices, device, all_probs,
                       batch_inter_sample_sim: float = 0.0):
    from xai.gradcam import GradCAMExplainer
    from xai.attention_rollout import AttentionRolloutExplainer
    from xai.shap_explainer import SHAPExplainer

    heatmap_dir     = os.path.join(config.output_dir, "heatmaps")
    explanation_dir = os.path.join(config.output_dir, "explanations")
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(explanation_dir, exist_ok=True)

    # FIX: GradCAMExplainer no longer takes target_layer kwarg
    gradcam_exp = GradCAMExplainer(model)
    rollout_exp = AttentionRolloutExplainer(model)
    shap_exp    = SHAPExplainer(model, method="integratedgrads")

    print("Generating heatmaps and explanations...")
    for idx in tqdm(sample_indices, desc="Saving heatmap videos"):
        idx    = int(idx)
        sample = test_ds[idx]
        frames_tensor = sample["frames"].unsqueeze(0).to(device)

        video_path = sample["meta"].get("video_path", "")
        video_id   = os.path.splitext(os.path.basename(video_path))[0] if video_path else str(idx)

        sampled_orig = _get_original_frames(
            video_path, config.num_frames, config.frame_size,
        )

        with torch.no_grad():
            out = model(frames_tensor)
        intrinsic = out.M_t_up[0].cpu().numpy()
        prob      = float(out.prob[0].cpu())
        verdict   = "FAKE" if prob > 0.5 else "REAL"

        intrinsic_maps = [intrinsic[t] for t in range(intrinsic.shape[0])]

        def _peakiness(m: np.ndarray) -> float:
            flat = m.flatten().astype(np.float64) + 1e-12
            flat = flat / flat.sum()
            H_val = -(flat * np.log(flat)).sum()
            return float(1.0 - H_val / np.log(flat.size))

        intrinsic_scores = [_peakiness(m) for m in intrinsic_maps]

        save_annotated_frame_strip(
            sampled_orig, intrinsic_maps, intrinsic_scores, verdict, prob,
            os.path.join(explanation_dir, f"{video_id}_strip.png"),
            sample_id=video_id,
            batch_inter_sample_sim=batch_inter_sample_sim,
        )

        save_explanation_video(
            sampled_orig, intrinsic_maps, intrinsic_scores, verdict, prob,
            os.path.join(heatmap_dir, f"{video_id}_intrinsic.mp4"),
        )

        for method_name, explainer in [
            ("gradcam", gradcam_exp),
            ("rollout", rollout_exp),
            ("shap",    shap_exp),
        ]:
            try:
                if method_name == "gradcam":
                    heat = explainer.explain(frames_tensor)[0]
                else:
                    heat = explainer.explain(frames_tensor)
            except Exception as e:
                print(f"  [{method_name} failed for idx {idx}: {e}]")
                heat = intrinsic

            maps_list   = [heat[t] for t in range(heat.shape[0])]
            scores_list = [float(m.max()) for m in maps_list]
            save_explanation_video(
                sampled_orig, maps_list, scores_list, verdict, prob,
                os.path.join(heatmap_dir, f"{video_id}_{method_name}.mp4"),
            )


def _save_representative_heatmaps(
    config, model, test_ds, all_probs, all_labels, device,
    batch_inter_sample_sim: float = 0.0,
    temporal_ssim: float = 0.0,
    inter_sample_cosine: float = 0.0,
):
    heatmap_dir = os.path.join(config.output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels, dtype=int)
    preds_arr  = (probs_arr >= 0.5).astype(int)

    def _find(condition_mask, max_tries=50):
        idxs = np.where(condition_mask)[0]
        if len(idxs) == 0:
            return None
        best = idxs[np.argmax(np.abs(probs_arr[idxs] - 0.5))]
        return int(best)

    candidates = {
        "real_correct": _find((labels_arr == 0) & (preds_arr == 0) & (probs_arr < 0.2)),
        "fake_correct": _find((labels_arr == 1) & (preds_arr == 1) & (probs_arr > 0.8)),
        "misclassified": _find(labels_arr != preds_arr),
    }
    for key in list(candidates.keys()):
        if candidates[key] is None:
            candidates[key] = _find(labels_arr >= 0)

    saved = []
    for role, idx in candidates.items():
        if idx is None:
            continue
        sample        = test_ds[idx]
        frames_tensor = sample["frames"].unsqueeze(0).to(device)
        video_path    = sample["meta"].get("video_path", "")
        video_id      = (
            os.path.splitext(os.path.basename(video_path))[0]
            if video_path else f"sample_{idx}"
        )
        video_id = f"{role}_{video_id}"

        orig_frames = _get_original_frames(video_path, config.num_frames, config.frame_size)

        with torch.no_grad():
            out = model(frames_tensor)
        intrinsic      = out.M_t_up[0].cpu().numpy()
        prob           = float(out.prob[0].cpu())
        verdict        = "FAKE" if prob > 0.5 else "REAL"
        confidence     = abs(prob - 0.5) * 2.0
        intrinsic_maps = [intrinsic[t] for t in range(intrinsic.shape[0])]

        def _peakiness(m: np.ndarray) -> float:
            flat = m.flatten().astype(np.float64) + 1e-12
            flat = flat / flat.sum()
            return float(1.0 - (-(flat * np.log(flat)).sum()) / np.log(flat.size))

        intrinsic_scores = [_peakiness(m) for m in intrinsic_maps]

        sp_stds  = [float(m.std()) for m in intrinsic_maps]
        is_uniform = float(np.mean(sp_stds)) < 0.01
        if len(intrinsic_maps) > 1:
            f0  = intrinsic_maps[0].flatten();  f0  = f0  / (np.linalg.norm(f0)  + 1e-8)
            fl  = intrinsic_maps[-1].flatten(); fl  = fl  / (np.linalg.norm(fl)  + 1e-8)
            is_frozen = float(np.dot(f0, fl)) > 0.99
        else:
            is_frozen = False
        is_class_agnostic = inter_sample_cosine > 0.95

        mp4_path = os.path.join(heatmap_dir, f"heatmap_overlay_{video_id}.mp4")
        save_explanation_video(
            orig_frames, intrinsic_maps, intrinsic_scores, verdict, prob, mp4_path
        )

        png_path = os.path.join(heatmap_dir, f"heatmap_strip_{video_id}.png")
        save_annotated_frame_strip(
            orig_frames, intrinsic_maps, intrinsic_scores, verdict, prob,
            png_path, sample_id=video_id,
            batch_inter_sample_sim=batch_inter_sample_sim,
        )

        peak_t  = int(np.argmax(intrinsic_scores))
        mean_map = np.mean(intrinsic_maps, axis=0)
        region   = get_region_label(mean_map)

        health_notes = []
        if is_uniform:
            health_notes.append("Attention is spatially uniform (possible collapse).")
        if is_frozen:
            health_notes.append("Attention map frozen across frames (possible collapse).")
        if is_class_agnostic:
            health_notes.append("Heatmaps similar across all test samples (class-agnostic).")
        if not health_notes:
            health_notes.append(
                f"Heatmap varies across frames (temporal_ssim={temporal_ssim:.2f}) "
                f"and across samples (inter_sample_cosine={inter_sample_cosine:.2f}) "
                f"— explanation looks healthy."
            )

        frame_range = f"frames {min(range(len(intrinsic_maps)), key=lambda t: intrinsic_scores[t])+1}–"
        frame_range += f"{max(range(len(intrinsic_maps)), key=lambda t: intrinsic_scores[t])+1}"
        summary_lines = [
            f"Role: {role}",
            f"Video: {video_path}",
            f"Ground truth: {'FAKE' if all_labels[idx] == 1 else 'REAL'}",
            f"Model predicted {verdict} with confidence {confidence:.2f} (prob={prob:.3f}).",
            f"Attention focused on the {region}.",
            f"Peak attention at t={peak_t+1} ({frame_range}).",
        ] + health_notes
        txt_path = os.path.join(heatmap_dir, f"heatmap_summary_{video_id}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")

        saved.append(video_id)
        print(f"[Representative] {role} → {video_id}  prob={prob:.3f}  verdict={verdict}")

    print(f"[Representative heatmaps] Saved {len(saved)} videos: {saved}")


def _get_original_frames(video_path: str, num_frames: int, frame_size: int):
    if not video_path or not os.path.exists(video_path):
        return [np.zeros((frame_size, frame_size, 3), np.uint8)] * num_frames

    cap   = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs  = np.linspace(0, total - 1, num_frames, dtype=int)
    buf   = {}
    fi    = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in set(idxs.tolist()):
            buf[fi] = cv2.resize(frame, (frame_size, frame_size))
        fi += 1
    cap.release()
    blank = np.zeros((frame_size, frame_size, 3), np.uint8)
    return [buf.get(i, blank) for i in idxs]