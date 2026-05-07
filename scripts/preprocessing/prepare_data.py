"""Prepare cleaned data and train/test split.

Declaration:
- This script is the py-first replacement for notebook-based preprocessing.
- It keeps the raw CSV untouched and writes cleaned artifacts to `data/`.

Tips:
- Run this first.
- Review the generated CSV files before training models.
"""

from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.data_utils import load_raw_data


def main():
    df = load_raw_data()
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()

    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = Path("data/interim/cleaned.csv")
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(cleaned_path, index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"cleaned: {cleaned_path}")
    print(f"train: {PROCESSED_DIR / 'train.csv'}")
    print(f"test: {PROCESSED_DIR / 'test.csv'}")


if __name__ == "__main__":
    main()
