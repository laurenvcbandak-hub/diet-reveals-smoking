"""Build the analysis cohort (adults 20+, reliable day-1 recall, known smoking status).

Saves outputs/cohort.csv and prints a cohort funnel.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def load(name):
    return pd.read_sas(f"data/{name}.xpt", format="xport", encoding="utf-8")


funnel = []


def step(label, df):
    funnel.append((label, len(df)))
    return df


# 1. All participants
demo = load("P_DEMO")
df = step("All participants (P_DEMO)", demo)

# 2. Adults 20+
df = step("Aged 20+", df[df["RIDAGEYR"] >= 20])

# 3. Reliable day-1 dietary recall
tot = load("P_DR1TOT")[["SEQN", "DR1DRSTZ", "WTDRD1PP", "DR1DAY"]]
df = df.merge(tot, on="SEQN", how="inner")
df = step("Reliable day-1 recall (DR1DRSTZ == 1, WTDRD1PP > 0)",
          df[(df["DR1DRSTZ"] == 1) & (df["WTDRD1PP"] > 0)])

# 4. Smoking status
smq = load("P_SMQ")[["SEQN", "SMQ020", "SMQ040", "SMD650", "SMQ050Q", "SMQ050U"]]
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
        "SDMVPSU", "SDMVSTRA", "WTDRD1PP", "DR1DAY", "smoking_status",
        "SMD650", "SMQ050Q", "SMQ050U"]
df = df[keep].copy()

# 6. Body measures and cotinine (left joins: missing values allowed)
bmx = load("P_BMX")[["SEQN", "BMXBMI", "BMXWAIST"]]
cot = load("P_COT")[["SEQN", "LBXCOT"]]
df = df.merge(bmx, on="SEQN", how="left").merge(cot, on="SEQN", how="left")

# 7. Recode refused / don't know codes to missing (per NHANES 2017-Mar 2020 codebook)
#   DMDEDUC2: 7 = refused, 9 = don't know
#   SMD650:   777 = refused, 999 = don't know (95 = "95 or more", kept)
#   SMQ050Q:  77777 = refused, 99999 = don't know; 66666 = "50 or more years" -> top-coded to 50
df["DMDEDUC2"] = df["DMDEDUC2"].replace({7: np.nan, 9: np.nan})
df["SMD650"] = df["SMD650"].replace({777: np.nan, 999: np.nan})
df["SMQ050Q"] = df["SMQ050Q"].replace({77777: np.nan, 99999: np.nan, 66666: 50})

df["SEQN"] = df["SEQN"].astype(int)

out = Path("outputs")
out.mkdir(exist_ok=True)
df.to_csv(out / "cohort.csv", index=False)

# Report
print("Cohort funnel")
for label, n in funnel:
    print(f"  {n:>6,}  {label}")
print()

print("Smoking status counts")
for status in ["current", "former", "never"]:
    print(f"  {status:<8} {(df['smoking_status'] == status).sum():>6,}")
print()

w = df["WTDRD1PP"]
pct_current = 100 * w[df["smoking_status"] == "current"].sum() / w.sum()
print(f"Survey-weighted % current smokers (WTDRD1PP): {pct_current:.1f}%")
print(f"\nSaved {len(df):,} rows x {df.shape[1]} columns to {out / 'cohort.csv'}")
