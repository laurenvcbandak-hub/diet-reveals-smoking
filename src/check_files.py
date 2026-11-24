"""Check the NHANES 2017-March 2020 files: size, unique SEQN, and key columns."""
import pandas as pd

files = ["P_DEMO", "P_SMQ", "P_DR1IFF", "P_DR1TOT", "P_BMX", "P_COT"]

expected_columns = {
    "P_SMQ": ["SMQ020", "SMQ040", "SMD650", "SMQ050Q", "SMQ050U"],
    "P_DR1TOT": ["DR1DRSTZ", "WTDRD1PP", "DR1DAY"],
    "P_DR1IFF": ["DR1_020", "DR1IFDCD", "DR1IKCAL", "DR1ICAFF"],
    "P_COT": ["LBXCOT"],
}

for f in files:
    df = pd.read_sas(f"data/{f}.xpt", format="xport", encoding="utf-8")
    print(f"== {f} ==")
    print(f"  rows:        {len(df):,}")
    print(f"  columns:     {df.shape[1]}")
    print(f"  unique SEQN: {df['SEQN'].nunique():,}")
    for col in expected_columns.get(f, []):
        status = "present" if col in df.columns else "MISSING"
        print(f"  {col:<10} {status}")
    print()
