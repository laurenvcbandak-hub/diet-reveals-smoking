import pandas as pd

files = [
    "P_DEMO",    # demographics: age, sex, race, income, survey weights
    "P_GHB",     # HbA1c
    "P_DIQ",     # diabetes questionnaire
    "P_SLQ",     # sleep: bedtime, snoring, breathing pauses, sleepiness
    "P_BMX",     # body measures: BMI, waist
    "P_SMQ",     # smoking
    "P_ALQ",     # alcohol
    "P_PAQ",     # physical activity
    "P_DR1TOT",  # day-1 diet recall: daily totals
    "P_DR1IFF",  # day-1 diet recall: each food item (large file, slow to load)
]

for f in files:
    df = pd.read_sas(f"data/{f}.xpt", encoding="infer")
    print(f, "columns:")
    print(list(df.columns))
    print()
