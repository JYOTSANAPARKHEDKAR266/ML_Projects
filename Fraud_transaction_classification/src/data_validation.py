import pandas as pd
from src.config import DATA_PATH, REPORTS_DIR, FIGURES_DIR, TARGET_COL
from src.utils import ensure_dirs

def main():
    ensure_dirs(REPORTS_DIR, FIGURES_DIR)

    df = pd.read_csv(DATA_PATH)

    n_rows, n_cols = df.shape
    missing = int(df.isna().sum().sum())
    dupes = int(df.duplicated().sum())
    class_counts = df[TARGET_COL].value_counts().to_dict()

    report = []
    report.append("# Data Validation Report\n")
    report.append(f"- File: `{DATA_PATH.name}`\n")
    report.append(f"- Shape: **{n_rows:,} rows × {n_cols} columns**\n")
    report.append(f"- Missing cells: **{missing:,}**\n")
    report.append(f"- Duplicate rows: **{dupes:,}**\n")
    report.append(f"- Class distribution: **{class_counts}**\n")
    report.append("\n## Columns\n")
    report.append(", ".join(df.columns) + "\n")

    out_path = REPORTS_DIR / "data_validation_report.md"
    out_path.write_text("".join(report), encoding="utf-8")
    print(f"✅ Wrote {out_path}")

if __name__ == "__main__":
    main()
