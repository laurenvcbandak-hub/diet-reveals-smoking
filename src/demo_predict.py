"""Fun interactive demo (NOT part of the main analysis).

Trains a small logistic regression on 8 easy diet questions (current vs never smokers,
NHANES 2017-March 2020), asks you the same questions and shows how "smoker-like" your
day of eating looks compared with US adults.

Usage:  python src/demo_predict.py
"""
import math
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
MG_PER_CUP = 95          # caffeine in one cup of brewed coffee / strong tea (approx.)
FEATURES = ["n_food_codes", "caffeine_mg", "first_caffeine_h", "no_caffeine", "any_soda_sugar",
            "any_beer", "any_fruit", "any_vegetables", "any_plain_water"]
LABELS = {"n_food_codes": "Number of different foods", "caffeine_mg": "Caffeine amount",
          "first_caffeine_h": "Time of first caffeine", "any_soda_sugar": "Sugary soda",
          "any_beer": "Beer", "any_fruit": "Fruit", "any_vegetables": "Vegetables",
          "any_plain_water": "Plain water"}

# ---------------------------------------------------------------------------
# 1. Train the small model
# ---------------------------------------------------------------------------
df = pd.read_csv("outputs/features.csv")
df = df[df["smoking_status"].isin(["current", "never"]) & df["kcal"].notna()].copy()
df["no_caffeine"] = df["first_caffeine_h"].isna().astype(int)
X, y = df[FEATURES], (df["smoking_status"] == "current").astype(int).to_numpy()

model = Pipeline([("imp", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                  ("clf", LogisticRegression(C=1.0, max_iter=5000))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
cv_auc = roc_auc_score(y, oof)
avg_never, avg_current = oof[y == 0].mean(), oof[y == 1].mean()
model.fit(X, y)

print("=" * 66)
print("  Smoker-like diet score - a fun demo")
print("=" * 66)
print(f"Model: logistic regression on 8 diet questions, trained on {len(y):,} US adults")
print(f"(NHANES 2017-2020, current vs never smokers). 5-fold CV AUC: {cv_auc:.3f}")
print("For comparison, the full 40-feature model + demographics reaches about 0.82.\n")


# ---------------------------------------------------------------------------
# 2. Questions
# ---------------------------------------------------------------------------
def ask(prompt, parse):
    while True:
        try:
            raw = input(prompt).strip().lstrip("﻿")   # piped input may start with a BOM
        except EOFError:
            sys.exit("\nNo answer given - exiting.")
        try:
            return parse(raw)
        except ValueError as e:
            print(f"  -> {e}")


def parse_count(s, lo, hi, what):
    if not s.replace(".", "", 1).isdigit():
        raise ValueError(f"please type a number for {what}")
    v = float(s)
    if not lo <= v <= hi:
        raise ValueError(f"please type a number between {lo} and {hi}")
    return v


def parse_yes_no(s):
    s = s.lower()
    if s in ("y", "yes"):
        return 1
    if s in ("n", "no"):
        return 0
    raise ValueError("please answer yes or no (y/n)")


def parse_time(s):
    if s.lower() in ("none", "no", "-", ""):
        return None
    try:
        h, m = s.split(":")
        h, m = int(h), int(m)
    except ValueError:
        raise ValueError("please use HH:MM (e.g. 07:30) or 'none'") from None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError("please use a valid time between 00:00 and 23:59")
    return h + m / 60


print("Think about everything you ate and drank yesterday (or today).\n")
n_foods = ask("1. How many different foods and drinks did you have? (e.g. 12): ",
              lambda s: parse_count(s, 0, 80, "foods"))
cups = ask("2. Cups of caffeinated coffee or tea? (e.g. 2): ",
           lambda s: parse_count(s, 0, 30, "cups"))
first_caf = None
if cups > 0:
    first_caf = ask("3. Time of your first caffeine (HH:MM, or 'none'): ", parse_time)
else:
    print("3. Time of first caffeine: skipped (no caffeine)")
answers = {
    "n_food_codes": n_foods,
    "caffeine_mg": cups * MG_PER_CUP,
    "first_caffeine_h": np.nan if first_caf is None else first_caf,
    "no_caffeine": int(first_caf is None),
    "any_soda_sugar": ask("4. Any sugary (non-diet) soda? (y/n): ", parse_yes_no),
    "any_beer": ask("5. Any beer? (y/n): ", parse_yes_no),
    "any_fruit": ask("6. Any whole fruit (not juice)? (y/n): ", parse_yes_no),
    "any_vegetables": ask("7. Any vegetables (not potatoes)? (y/n): ", parse_yes_no),
    "any_plain_water": ask("8. Any plain water? (y/n): ", parse_yes_no),
}

# ---------------------------------------------------------------------------
# 3. Score and what pushed it up or down
# ---------------------------------------------------------------------------
person = pd.DataFrame([answers])[FEATURES]
score = model.predict_proba(person)[0, 1]

# Contribution of each answer (log-odds) relative to an average person in the data:
# coefficient x standardized value. Caffeine timing and the no-caffeine flag are combined.
z = model.named_steps["s"].transform(model.named_steps["imp"].transform(person))[0]
contrib = dict(zip(FEATURES, model.named_steps["clf"].coef_[0] * z))
contrib["first_caffeine_h"] += contrib.pop("no_caffeine")


def bar(p, width=30):
    filled = round(p * width)
    return "#" * filled + "." * (width - filled)


print("\n" + "=" * 66)
print(f"  Your smoker-like diet score: {100 * score:5.1f}%   [{bar(score)}]")
print("=" * 66)
print(f"  Average never smoker:        {100 * avg_never:5.1f}%   [{bar(avg_never)}]")
print(f"  Average current smoker:      {100 * avg_current:5.1f}%   [{bar(avg_current)}]")

def shown(f):
    v = answers[f]
    if f == "n_food_codes":
        return f"{v:g}"
    if f == "caffeine_mg":
        return f"{cups:g} cups = {v:.0f} mg"
    if f == "first_caffeine_h":
        return "none" if first_caf is None else f"{int(v):02d}:{round(v % 1 * 60):02d}"
    return "yes" if v else "no"


print("\nWhat moved your score (compared with an average US adult in the data):")
for f, c in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
    if abs(c) < 0.02:
        effect = "no real effect"
    else:
        strength = "strongly " if abs(c) >= 0.3 else ""
        effect = f"pushed {strength}{'UP' if c > 0 else 'down'}"
    print(f"  {'+' if c > 0 else '-'} {LABELS[f] + ': ' + shown(f):<38} {effect}")

# ---------------------------------------------------------------------------
# 4. Disclaimer
# ---------------------------------------------------------------------------
print("\n" + "-" * 66)
print("NOTE: This is a statistical demo based on US adults (NHANES 2017-2020).")
print("The score describes how typical a diet pattern is among smokers vs")
print("never smokers. It CANNOT tell whether a person smokes: many smokers")
print("eat 'never-smoker' diets and many never smokers eat 'smoker' diets.")
print("-" * 66)
