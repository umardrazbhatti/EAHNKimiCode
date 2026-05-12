"""
utils/visualization.py — Explanation visualization utilities.
"""

import os
import cv2
import numpy as np


def overlay_heatmap_on_frame(
    frame_bgr: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
):
    H, W = frame_bgr.shape[:2]

    attn_resized = cv2.resize(
        attention_map.astype(np.float32), (W, H),
        interpolation=cv2.INTER_LINEAR,
    )

    a_min, a_max = attn_resized.min(), attn_resized.max()
    attn_norm = (attn_resized - a_min) / (a_max - a_min + 1e-8)

    heatmap_u8  = (attn_norm * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, colormap)
    overlay     = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap_bgr, alpha, 0)

    binary    = (attn_norm >= 0.6).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = max(y - 5, 12)
        cv2.putText(
            overlay, "High Attention", (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
        )

    return overlay, attn_norm


_REGION_LABELS = {
    ("upper",  "left"):   "upper-left periocular region",
    ("upper",  "center"): "upper-central forehead and brow region",
    ("upper",  "right"):  "upper-right periocular region",
    ("middle", "left"):   "left cheek and ear region",
    ("middle", "center"): "central nasal and mid-face region",
    ("middle", "right"):  "right cheek and ear region",
    ("lower",  "left"):   "lower-left jaw and mouth region",
    ("lower",  "center"): "lower-central mouth and chin region",
    ("lower",  "right"):  "lower-right jaw and mouth region",
}


def get_region_label(attn_map: np.ndarray) -> str:
    peak_idx = int(np.argmax(attn_map))
    r, c = np.unravel_index(peak_idx, attn_map.shape)
    H, W = attn_map.shape

    row_frac = r / max(H - 1, 1)
    if row_frac < 2 / 6:
        row_key = "upper"
    elif row_frac <= 4 / 6:
        row_key = "middle"
    else:
        row_key = "lower"

    col_frac = c / max(W - 1, 1)
    if col_frac < 2 / 6:
        col_key = "left"
    elif col_frac <= 4 / 6:
        col_key = "center"
    else:
        col_key = "right"

    label = _REGION_LABELS.get((row_key, col_key), "central nasal and mid-face region")
    return f"{label} (peak at row={r}, col={c})"


def generate_explanation_text(
    verdict: str,
    confidence: float,
    prob: float,
    attention_scores: list,
    attention_maps: list,
    batch_inter_sample_sim: float = 0.0,
) -> str:
    T = len(attention_scores)
    M_t_up = np.stack(attention_maps) if T > 0 else np.zeros((1, 7, 7))

    spatial_std_per_frame = [float(m.std()) for m in M_t_up]
    is_spatially_uniform  = float(np.mean(spatial_std_per_frame)) < 0.01

    if T > 1:
        flat0     = M_t_up[0].flatten() / (np.linalg.norm(M_t_up[0]) + 1e-8)
        flat_last = M_t_up[-1].flatten() / (np.linalg.norm(M_t_up[-1]) + 1e-8)
        is_temporally_frozen = float(np.dot(flat0, flat_last)) > 0.99
    else:
        is_temporally_frozen = False

    is_class_agnostic = batch_inter_sample_sim > 0.95

    sorted_frames = sorted(range(T), key=lambda i: attention_scores[i], reverse=True)
    top3          = sorted_frames[:3]

    lines = [
        f"VERDICT: This video is likely {verdict} (confidence: {confidence:.0%}).",
        "",
        "EXPLANATION:",
    ]

    if is_spatially_uniform and is_temporally_frozen and is_class_agnostic:
        lines.append(
            "  ⚠ EXPLANATION COLLAPSE DETECTED — heatmap is identical across frames "
            "AND across samples. Re-train with stronger inter-sample diversity loss."
        )
    elif is_spatially_uniform:
        lines.append(
            "  • Attention is spatially uniform within frames (no localised focus). "
            "The explanation head may need stronger diversity regularisation."
        )
    elif is_temporally_frozen:
        lines.append(
            "  • Attention map is nearly identical across all frames (temporally frozen). "
            "Consider reducing lambda2 (temporal consistency weight)."
        )
    elif is_class_agnostic:
        lines.append(
            "  • Attention maps are very similar across samples in this batch "
            "(class-agnostic). The diversity loss may need increasing."
        )
    else:
        top3_labels = ", ".join(str(f + 1) for f in top3)
        lines.append(f"  • Attention was highest in frames {top3_labels}.")

    mean_attn = np.mean(M_t_up, axis=0)
    region    = get_region_label(mean_attn)
    lines.append(f"  • The primary area of concern is the {region}.")

    if verdict == "FAKE":
        lines.append("  • High attention in this area may indicate:")
        lines.append("      - Blending boundary artifacts at face-swap seams")
        lines.append("      - Unnatural skin texture or colour inconsistencies")
        lines.append("      - Identity inconsistencies introduced by face-swap methods")
        lines.append("      - GAN frequency fingerprints in shallow texture layers")
    else:
        lines.append("  • No strong manipulation artifacts were detected.")
        lines.append(
            "    Facial regions show consistent texture and identity across frames."
        )

    lines.append("")
    lines.append("ATTENTION SCORES PER FRAME:")
    for i, score in enumerate(attention_scores):
        filled = int(score * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        lines.append(f"  Frame {i + 1:02d}: [{bar}]  {score:.3f}")

    return "\n".join(lines)


def save_annotated_frame_strip(
    frames_bgr: list,
    attention_maps: list,
    attention_scores: list,
    verdict: str,
    prob: float,
    output_path: str,
    sample_id: str,
    batch_inter_sample_sim: float = 0.0,
) -> str:
    from PIL import Image as PILImage, ImageDraw, ImageFont

    T        = len(frames_bgr)
    n_select = min(T, 8)
    sel_idx  = np.linspace(0, T - 1, n_select, dtype=int)

    annotated_frames = []
    for idx in sel_idx:
        frame   = cv2.resize(frames_bgr[idx], (224, 224))
        overlay, _ = overlay_heatmap_on_frame(frame, attention_maps[idx])
        label   = f"F{idx + 1:02d}  attn:{attention_scores[idx]:.2f}"
        cv2.putText(
            overlay, label, (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
        )
        annotated_frames.append(overlay)

    strip   = np.hstack(annotated_frames)
    strip_w = strip.shape[1]

    confidence = prob if prob >= 0.5 else (1.0 - prob)
    text       = generate_explanation_text(
        verdict, confidence, prob, attention_scores, attention_maps,
        batch_inter_sample_sim=batch_inter_sample_sim,
    )
    text_lines = text.split("\n")
    line_h     = 17
    top_margin = 10
    left_margin = 10
    panel_h    = len(text_lines) * line_h + 20

    panel_pil = PILImage.new("RGB", (strip_w, panel_h), (20, 20, 20))
    draw      = ImageDraw.Draw(panel_pil)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13
        )
    except Exception:
        font = ImageFont.load_default()

    verdict_color = (255, 80, 80) if verdict == "FAKE" else (80, 255, 80)
    other_color   = (220, 220, 220)

    for i, line in enumerate(text_lines):
        y     = top_margin + i * line_h
        color = verdict_color if i == 0 else other_color
        draw.text((left_margin, y), line, fill=color, font=font)

    panel_bgr   = cv2.cvtColor(np.array(panel_pil), cv2.COLOR_RGB2BGR)
    final_image = np.vstack([strip, panel_bgr])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, final_image)

    txt_path = output_path.replace(".png", "_explanation.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return output_path


def save_explanation_video(
    frames_bgr: list,
    attention_maps: list,
    attention_scores: list,
    verdict: str,
    prob: float,
    output_path: str,
    fps: int = 5,
) -> None:
    T          = len(frames_bgr)
    confidence = prob if prob >= 0.5 else (1.0 - prob)
    verdict_color_bgr = (80, 80, 255) if verdict == "FAKE" else (80, 255, 80)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (224, 224 + 80))

    for t in range(T):
        frame   = cv2.resize(frames_bgr[t], (224, 224))
        overlay, _ = overlay_heatmap_on_frame(frame, attention_maps[t])

        panel = np.full((80, 224, 3), 20, dtype=np.uint8)

        cv2.putText(
            panel, f"{verdict} ({confidence:.0%} conf)", (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, verdict_color_bgr, 1,
        )
        region = get_region_label(attention_maps[t])
        cv2.putText(
            panel, f"Frame {t + 1:02d}/{T} | Region: {region}", (6, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
        )
        cv2.putText(
            panel, f"Attn: {attention_scores[t]:.3f}", (6, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )
        bar_max_w = 224 - 80
        bar_w     = int(attention_scores[t] * bar_max_w)
        cv2.rectangle(panel, (8, 58), (8 + bar_max_w, 70), (100, 100, 100), 1)
        if bar_w > 0:
            cv2.rectangle(panel, (8, 58), (8 + bar_w, 70), (100, 200, 255), -1)

        combined = np.vstack([overlay, panel])
        writer.write(combined)

    writer.release()
