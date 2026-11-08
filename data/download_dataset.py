"""Download the official UCI Bank Marketing dataset and save bank-full.csv."""
from pathlib import Path
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bank-full.csv"

def main():
    print("Fetching UCI Bank Marketing dataset (dataset id 222)...")
    dataset = fetch_ucirepo(id=222)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    # UCI's legacy bank-full variant has 45,211 rows and 16 predictors.
    frame = X.copy()
    target_name = y.columns[0]
    frame["y"] = y[target_name].astype(str).str.lower().str.strip()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"Saved {len(frame):,} rows and {frame.shape[1]-1} predictors to {OUT}")
    print("Target distribution:")
    print(frame["y"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
