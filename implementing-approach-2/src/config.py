"""Shared config for Approach-2 implementation (tournament core)."""
import os
from pathlib import Path

HERE = Path.cwd()
if (HERE / "src").exists():
    ROOT = HERE
elif (HERE / "implementing-approach-2" / "src").exists():
    ROOT = HERE / "implementing-approach-2"
else:
    ROOT = HERE
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

ENV_PROC = os.environ.get("PROCESSED_ROOT", "").strip()
if ENV_PROC:
    PROC = Path(ENV_PROC)
else:
    cands = [Path("C:/Games/ML-2026/Amazon-ML-Challenge-2026/processed"),
             ROOT.parent / "Amazon-ML-Challenge-2026" / "processed",
             Path("/kaggle/input/approach2-processed")]
    PROC = next((p for p in cands if p.exists()), cands[0])

ENV_DATA = os.environ.get("DATA_ROOT", "").strip()
if ENV_DATA:
    DATA = Path(ENV_DATA)
else:
    cands = [Path("C:/Games/ML-2026/6ab10eb3b23ba_student_resource/student_resource/dataset"),
             ROOT.parent / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"]
    DATA = next((p for p in cands if p.exists()), cands[0])

GT = DATA / "train" / "train_ground_truth.tsv"
SEED = 42
COUNTRIES_TRAIN = ["US", "India"]
DEMO_S1 = 8000          # train S1 per country for the demo (FULL: all rows)
TFIDF_TOPK = 15
TFIDF_NGRAMS = (2, 5)
THRESHOLDS = [round(x, 2) for x in
              [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
               0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]
LGBM_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
               "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
               "bagging_freq": 1, "lambda_l2": 1.0, "verbosity": -1, "seed": SEED}
LGBM_ROUNDS = 500
LGBM_EARLY = 50
NEG_RATIO = 0  # 0 = use all retrieved negatives (already hard); cap per-S1 if >0
