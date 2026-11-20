import pandas as pd

files = ["P_DEMO", "P_SMQ", "P_BMX", "P_DR1TOT", "P_DR1IFF"]
for f in files:
    df = pd.read_sas(f"data/{f}.xpt")
    print(f, "→", df.shape[0], "rows,", df.shape[1], "columns")
    