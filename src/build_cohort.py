"""Build the analysis cohort (adults 20+, reliable day-1 recall, known smoking status).

Run as a script: builds the 2017-March 2020 cohort, saves outputs/cohort.csv and prints a
cohort funnel. `build_cohort()` is reused for other NHANES cycles (external_validation.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd

# File names per cycle. `weight` is the day-1 dietary weight, whose name differs by cycle.
CYCLE_2017 = {"demo": "data/P_DEMO.xpt", "tot": "data/P_DR1TOT.xpt", "smq": "data/P_SMQ.xpt",
              "bmx": "data/P_BMX.xpt", "cot": "data/P_COT.xpt", "weight": "WTDRD1PP"}


def load(path):
    return pd.read_sas(path, format="xport", encoding="utf-8")


def build_cohort(files):
    """Apply the cohort rules to one NHANES cycle. Returns (cohort DataFrame, funnel list).

    files: dict with paths for demo, tot, smq, bmx, cot (cot may be None) and the name of
    the day-1 dietary weight column (`weight`).
    """
    w = files["weight"]
    funnel = []

    def step(label, df):
        funnel.append((label, len(df)))
        return df

    # 1. All participants
    demo = load(files["demo"])
    df = step("All participants (DEMO)", demo)

    # 2. Adults 20+
    df = step("Aged 20+", df[df["RIDAGEYR"] >= 20])

    # 3. Reliable day-1 dietary recall
    tot = load(files["tot"])[["SEQN", "DR1DRSTZ", w, "DR1DAY"]]
    df = df.merge(tot, on="SEQN", how="inner")
    df = step(f"Reliable day-1 recall (DR1DRSTZ == 1, {w} > 0)",
              df[(df["DR1DRSTZ"] == 1) & (df[w] > 0)])

    # 4. Smoking status
    smq = load(files["smq"])[["SEQN", "SMQ020", "SMQ040", "SMD650", "SMQ050Q", "SMQ050U"]]
    df = df.merge(smq, on="SEQN", how="inner")
    conditions = [
        (df["SMQ020"] == 1) & df["SMQ040"].isin([1, 2]),
        (df["SMQ020"] == 1) & (df["SMQ040"] == 3),
        (df["SMQ020"] == 2),
    ]
    df["smoking_status"] = np.select(conditions, ["current", "former", "never"], default=None)
    df = step("Known smoking status (current / former / never)", df[df["smoking_status"].notna()])

    # 5. Keep analysis columns
    keep = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
            "SDMVPSU", "SDMVSTRA", w, "DR1DAY", "smoking_status",
            "SMD650", "SMQ050Q", "SMQ050U"]
    df = df[keep].copy()

    # 6. Body measures and cotinine (left joins: missing values allowed)
    bmx = load(files["bmx"])[["SEQN", "BMXBMI", "BMXWAIST"]]
    df = df.merge(bmx, on="SEQN", how="left")
    if files.get("cot"):
        cot = load(files["cot"])[["SEQN", "LBXCOT"]]
        df = df.merge(cot, on="SEQN", how="left")
    else:
        df["LBXCOT"] = np.nan

    # 7. Recode refused / don't know codes to missing (per NHANES codebooks)
    #   DMDEDUC2: 7 = refused, 9 = don't know
    #   SMD650:   777 = refused, 999 = don't know (95 = "95 or more", kept)
    #   SMQ050Q:  77777 = refused, 99999 = don't know; 66666 = "50 or more years" -> top-coded to 50
    df["DMDEDUC2"] = df["DMDEDUC2"].replace({7: np.nan, 9: np.nan})
    df["SMD650"] = df["SMD650"].replace({777: np.nan, 999: np.nan})
    df["SMQ050Q"] = df["SMQ050Q"].replace({77777: np.nan, 99999: np.nan, 66666: 50})

    df["SEQN"] = df["SEQN"].astype(int)
    return df, funnel


def print_funnel(funnel):
    print("Cohort funnel")
    for label, n in funnel:
        print(f"  {n:>6,}  {label}")
    print()


if __name__ == "__main__":
    df, funnel = build_cohort(CYCLE_2017)

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "cohort.csv", index=False)

    print_funnel(funnel)

    print("Smoking status counts")
    for status in ["current", "former", "never"]:
        print(f"  {status:<8} {(df['smoking_status'] == status).sum():>6,}")
    print()

    w = df[CYCLE_2017["weight"]]
    pct_current = 100 * w[df["smoking_status"] == "current"].sum() / w.sum()
    print(f"Survey-weighted % current smokers (WTDRD1PP): {pct_current:.1f}%")
    print(f"\nSaved {len(df):,} rows x {df.shape[1]} columns to {out / 'cohort.csv'}")
