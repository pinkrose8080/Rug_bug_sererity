"""
===============================================================================
Data Preparation & Integrity Validation for Bug Severity Classification
===============================================================================
Validates clean historical and holdout test partitions:
- Historical Data: 2,306 samples (balanced memory)
- Test Data: 576 samples (278 Severe, 298 Non-Severe)
- Leakage Check: 0 bug ID overlap, fixed few-shot exemplars strictly excluded from test set.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN_FILE = DATA / "historical_data_clean.csv"
TEST_FILE = DATA / "test_data_clean.csv"
FEW_SHOT_BUG_IDS = {"1739372", "1700486", "1797139"}


def validate_and_prepare() -> None:
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Expected clean datasets at {TRAIN_FILE} and {TEST_FILE}. Please ensure data files are present."
        )

    train = pd.read_csv(TRAIN_FILE, dtype={"bug_id": str})
    test = pd.read_csv(TEST_FILE, dtype={"bug_id": str})

    # Integrity verification: zero overlap
    overlap = set(train["bug_id"]) & set(test["bug_id"])
    if overlap:
        raise AssertionError(f"Data leakage detected! {len(overlap)} shared bug IDs between train and test.")

    few_shot_in_test = set(test["bug_id"]) & FEW_SHOT_BUG_IDS
    if few_shot_in_test:
        raise AssertionError(f"Few-shot examples found in test set: {few_shot_in_test}")

    print("=" * 60)
    print(" [OK] Dataset Integrity Verified (Leak-Free Holdout)")
    print("=" * 60)
    print(f" Train Partition (Historical Memory): {len(train):,} reports")
    print(f"   - Class Distribution: {train['severity_label'].value_counts().to_dict()}")
    print(f" Test Partition (Holdout Evaluation): {len(test):,} reports")
    print(f"   - Class Distribution: {test['severity_label'].value_counts().to_dict()}")
    print(f" Shared Bug IDs (Overlap): {len(overlap)}")
    print(f" Few-Shot Exemplars in Test: {len(few_shot_in_test)}")
    print("=" * 60)


if __name__ == "__main__":
    validate_and_prepare()
