"""
run_full_pipeline.py — Entry point for Kaggle / local execution.
"""

import os
from config import EAHNConfig, parse_args
from scripts.train_real import main as train_main
from scripts.dashboard import show_dashboard


def main():
    args   = parse_args()
    config = EAHNConfig.from_args(args)
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Output directory: {config.output_dir}")
    print(f"Device: {config.device}")
    print(f"Dataset: {config.dataset_name}")
    train_main(config)
    show_dashboard(config.output_dir)
    print("Full pipeline completed. Outputs in", config.output_dir)


if __name__ == "__main__":
    main()
