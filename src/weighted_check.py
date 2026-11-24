"""Sensitivity analysis: AUCs weighted with the day-1 dietary survey weight (WTDRD1PP).

Out-of-fold predictions (5-fold, current vs never smokers) for models A, C and E (HGB);
AUC unweighted vs weighted, and weighted delta AUC (E - A) with a bootstrap CI.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
N_BOOT = 1000
WEIGHT = "WTDRD1PP"

# ---------------------------------------------------------------------------
# Data (same preparation as train_models.py)
# ---------------------------------------------------------------------------
df = pd.read_csv("outputs/features.csv")
cohort_cols = pd.read_csv("outputs/cohort.csv", nrows=0).columns
diet_cols = [c for c in df.columns if c not in cohort_cols]
df = df[df["smoking_status"].isin(["current", "never"]) & df["kcal"].notna()].reset_index(drop=True)
df["no_caffeine"] = df["first_caffeine_h"].isna().astype(int)
y = (df["smoking_status"] == "current").astype(int).to_numpy()
w = df[WEIGHT].to_numpy()

DEMO_NUM, DEMO_CAT, DEMO_IMPUTE = ["RIDAGEYR"], ["RIAGENDR", "RIDRETH3", "DMDEDUC2"], ["INDFMPIR"]
BODY = ["BMXBMI", "BMXWAIST"]
DIET = diet_cols + ["no_caffeine"]
X = df[DEMO_NUM + DEMO_CAT + DEMO_IMPUTE + BODY + DIET]


def preprocessor(demo, body, diet, scale):
    tail = [("s", StandardScaler())] if scale else []
    blocks = []
    if demo:
        blocks += [
            ("age", Pipeline([("pass", "passthrough")] + tail), DEMO_NUM),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                              ("oh", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]),
             DEMO_CAT),
            ("pir", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))] + tail),
             DEMO_IMPUTE),
        ]
    if body:
        blocks.append(("body", Pipeline([("imp", SimpleImputer(strategy="median",
                                                               add_indicator=True))] + tail), BODY))
    if diet:
        blocks.append(("diet", Pipeline([("imp", SimpleImputer(strategy="median"))] + tail), DIET))
    return ColumnTransformer(blocks)


def make_model(demo, body, diet, method):
    if method == "logreg":
        return Pipeline([("prep", preprocessor(demo, body, diet, True)),
                         ("clf", LogisticRegression(C=1.0, max_iter=5000))])
    return Pipeline([("prep", preprocessor(demo, body, diet, False)),
                     ("clf", HistGradientBoostingClassifier(random_state=SEED))])


MODELS = {"A  demographics (logreg)": (True, False, False, "logreg"),
          "C  diet only (logreg)": (False, False, True, "logreg"),
          "E  demo+body+diet (HGB)": (True, True, True, "hgb")}

# ---------------------------------------------------------------------------
# 1. Out-of-fold predictions (models fit unweighted, as in train_models.py)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {name: np.zeros(len(y)) for name in MODELS}
for tr, va in cv.split(X, y):
    for name, spec in MODELS.items():
        oof[name][va] = make_model(*spec).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]

# ---------------------------------------------------------------------------
# 2 + 4. Unweighted vs weighted AUC
# ---------------------------------------------------------------------------
line = "=" * 64
print(f"Current + never smokers: n = {len(y):,} ({y.sum():,} current); "
      f"5-fold out-of-fold predictions\n")
print(line + f"\nAUC, unweighted vs weighted ({WEIGHT})\n" + line)
print(f"{'Model':<28}{'Unweighted':>12}{'Weighted':>12}{'Difference':>12}")
auc = {}
for name, p in oof.items():
    u, wt = roc_auc_score(y, p), roc_auc_score(y, p, sample_weight=w)
    auc[name] = (u, wt)
    print(f"{name:<28}{u:>12.3f}{wt:>12.3f}{wt - u:>+12.3f}")

# ---------------------------------------------------------------------------
# 3. Weighted delta AUC (E - A), bootstrap over people keeping their weights
# ---------------------------------------------------------------------------
pA, pE = oof["A  demographics (logreg)"], oof["E  demo+body+diet (HGB)"]
rng = np.random.default_rng(SEED)
deltas_w, deltas_u = [], []
while len(deltas_w) < N_BOOT:
    i = rng.integers(0, len(y), len(y))
    if 0 < y[i].sum() < len(i):
        deltas_w.append(roc_auc_score(y[i], pE[i], sample_weight=w[i])
                        - roc_auc_score(y[i], pA[i], sample_weight=w[i]))
        deltas_u.append(roc_auc_score(y[i], pE[i]) - roc_auc_score(y[i], pA[i]))
d_w = auc["E  demo+body+diet (HGB)"][1] - auc["A  demographics (logreg)"][1]
d_u = auc["E  demo+body+diet (HGB)"][0] - auc["A  demographics (logreg)"][0]
lo_w, hi_w = np.percentile(deltas_w, [2.5, 97.5])
lo_u, hi_u = np.percentile(deltas_u, [2.5, 97.5])
print(f"\nDelta AUC E - A ({N_BOOT} paired bootstraps of people, weights kept):")
print(f"  weighted:   {d_w:+.3f}  [{lo_w:+.3f}, {hi_w:+.3f}]")
print(f"  unweighted: {d_u:+.3f}  [{lo_u:+.3f}, {hi_u:+.3f}]   (for reference)")

# ---------------------------------------------------------------------------
# 5. Survey-weighted prevalence
# ---------------------------------------------------------------------------
pct_w = 100 * w[y == 1].sum() / w.sum()
print(f"\nCurrent smokers among current + never smokers: "
      f"unweighted {100 * y.mean():.1f}%, survey-weighted ({WEIGHT}) {pct_w:.1f}%")
