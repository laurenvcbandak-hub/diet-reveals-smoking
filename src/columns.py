import pandas as pd

for f in ["P_GHB", "P_SLQ", "P_DIQ"]:
    df = pd.read_sas(f"data/{f}.xpt")
    print(f, "columns:")
    print(list(df.columns))
    print()
    