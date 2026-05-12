"""
scripts/data_analysis.py — FF++ c23 dataset statistics.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


FF_METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]


def _count_mp4(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() == ".mp4")


def _probe_video(path: Path) -> dict:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "json", str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data   = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        return {
            "width":    stream.get("width"),
            "height":   stream.get("height"),
            "duration": stream.get("duration"),
        }
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="FF++ c23 dataset analysis")
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--dataset_name", default="ff++")
    parser.add_argument("--output_dir", default="outputs/")
    args = parser.parse_args()

    if args.dataset_name == "celeb_df":
        print("Celeb-DF v2 deferred — see future work.")
        sys.exit(0)

    root = Path(args.data_root)
    if not root.exists():
        print(f"ERROR: data_root not found: {root}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    real_dir   = root / "original_sequences" / "youtube" / "c23" / "videos"
    n_real     = _count_mp4(real_dir)

    fake_counts = {}
    for method in FF_METHODS:
        d = root / "manipulated_sequences" / method / "c23" / "videos"
        fake_counts[method] = _count_mp4(d)
    n_fake = sum(fake_counts.values())

    n_total = n_real + n_fake
    ratio   = n_fake / max(n_real, 1)

    print("=" * 60)
    print("FF++ c23 Dataset Analysis")
    print("=" * 60)
    print(f"  data_root : {root}")
    print(f"  real      : {n_real:>6}  ({real_dir.relative_to(root)})")
    for method, cnt in fake_counts.items():
        print(f"  {method:<20}: {cnt:>6}")
    print(f"  {'TOTAL fake':<20}: {n_fake:>6}")
    print(f"  {'TOTAL':<20}: {n_total:>6}  ratio={ratio:.2f}:1 (fake:real)")
    print()

    from sklearn.model_selection import train_test_split
    real_samples = [{"label": 0}] * n_real
    fake_samples = [{"label": 1}] * n_fake
    all_samples  = real_samples + fake_samples
    labels       = [s["label"] for s in all_samples]

    train_val, test = train_test_split(
        all_samples, test_size=0.1, stratify=labels, random_state=42
    )
    tv_labels = [s["label"] for s in train_val]
    train, val = train_test_split(
        train_val, test_size=0.1 / 0.9, stratify=tv_labels, random_state=42
    )

    split_info = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        nr = sum(1 for s in split if s["label"] == 0)
        nf = sum(1 for s in split if s["label"] == 1)
        split_info[name] = {"total": len(split), "real": nr, "fake": nf}
        print(f"  {name:<6}: total={len(split):>5}  real={nr:>5}  fake={nf:>5}")
    print()

    sample_props = []
    videos_to_probe = list(real_dir.glob("*.mp4"))[:5]
    for v in videos_to_probe:
        info = _probe_video(v)
        info["path"] = str(v.name)
        sample_props.append(info)
        print(f"  Sample: {v.name}  {info}")

    result = {
        "dataset":      "ff++ c23",
        "data_root":    str(root),
        "n_real":       n_real,
        "n_fake":       n_fake,
        "n_total":      n_total,
        "ratio":        round(ratio, 2),
        "fake_by_method": fake_counts,
        "splits":       split_info,
        "sample_videos": sample_props,
    }
    out_path = os.path.join(args.output_dir, "data_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved analysis to {out_path}")


if __name__ == "__main__":
    main()
