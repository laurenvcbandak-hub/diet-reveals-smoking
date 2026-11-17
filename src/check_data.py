import pandas as pd

files = ["P_DR1IFF", "P_GHB", "P_SLQ", "P_DIQ", "P_DEMO"]
for f in files:
    df = pd.read_sas(f"data/{f}.xpt")
    print(f, "→", df.shape[0], "rows,", df.shape[1], "columns")
    