"""Sensitivity analysis: restrict to cotinine-confirmed current and never smokers.

Serum cotinine (LBXCOT, ng/mL) cutoffs: 3 ng/mL (Benowitz et al. 2009) and 10 ng/mL.
Repeats the main comparison (A vs C vs E, delta AUC E - A) in each confirmed sample.
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

# ---------------------------------------------------------------------------
# Data (same preparation as train_models.py)
# ---------------------------------------------------------------------------
df = pd.read_csv("outputs/features.csv")
cohort_cols = pd.read_csv("outputs/cohort.csv", nrows=0).columns
diet_cols = [c for c in df.columns if c not in cohort_cols]
df = df[df["smoking_status"].isin(["current", "never"]) & df["kcal"].notna()].reset_index(drop=True)
df["no_caffeine"] = df["first_caffeine_h"].isna().astype(int)

DEMO_NUM, DEMO_CAT, DEMO_IMPUTE = ["RIDAGEYR"], ["RIAGENDR", "RIDRETH3", "DMDEDUC2"], ["INDFMPIR"]
BODY = ["BMXBMI", "BMXWAIST"]
DIET = diet_cols + ["no_caffeine"]
FEATURES = DEMO_NUM + DEMO_CAT + DEMO_IMPUTE + BODY + DIET     # LBXCOT is never a predictor


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


MODELS = {"A": (True, False, False, "logreg"),     # demographics, logreg
          "C": (False, False, True, "logreg"),     # diet only, logreg
          "E": (True, True, True, "hgb")}          # demo + body + diet, HGB


def evaluate(data):
    """5-fold CV AUC for A, C, E and paired-bootstrap delta AUC (E - A) on OOF predictions."""
    X, y = data[FEATURES], (data["smoking_status"] == "current").astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {k: np.zeros(len(y)) for k in MODELS}
    fold_auc = {k: [] for k in MODELS}
    for tr, va in cv.split(X, y):
        for k, spec in MODELS.items():
            p = make_model(*spec).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
            oof[k][va] = p
            fold_auc[k].append(roc_auc_score(y[va], p))
    rng = np.random.default_rng(SEED)
    deltas = []
    while len(deltas) < N_BOOT:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            deltas.append(roc_auc_score(y[i], oof["E"][i]) - roc_auc_score(y[i], oof["A"][i]))
    out = {"n": len(y), "n_current": int(y.sum())}
    for k in MODELS:
        out[f"cv_{k}"] = np.mean(fold_auc[k])
        out[f"cv_{k}_sd"] = np.std(fold_auc[k])
    out["delta_EA"] = roc_auc_score(y, oof["E"]) - roc_auc_score(y, oof["A"])
    out["delta_lo"], out["delta_hi"] = np.percentile(deltas, [2.5, 97.5])
    return out


def print_eval(r):
    print(f"  n = {r['n']:,} ({r['n_current']:,} current, {r['n_current'] / r['n']:.1%})")
    for k, lab in [("A", "A demo (logreg)"), ("C", "C diet (logreg)"), ("E", "E demo+body+diet (HGB)")]:
        print(f"  {lab:<26} CV AUC {r[f'cv_{k}']:.3f} (SD {r[f'cv_{k}_sd']:.3f})")
    print(f"  Delta AUC E - A (OOF, {N_BOOT} paired bootstraps): "
          f"{r['delta_EA']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]")


line = "=" * 70

# ---------------------------------------------------------------------------
# 1. Cotinine availability and distribution
# ---------------------------------------------------------------------------
has_cot = df["LBXCOT"].notna()
print(line + "\n1. Serum cotinine (ng/mL) among current and never smokers\n" + line)
print(f"With cotinine: {has_cot.sum():,} of {len(df):,} ({has_cot.mean():.1%})")
for s in ["never", "current"]:
    c = df.loc[(df["smoking_status"] == s) & has_cot, "LBXCOT"]
    n_all = (df["smoking_status"] == s).sum()
    q1, med, q3 = c.quantile([0.25, 0.5, 0.75])
    print(f"  {s:<8} n = {len(c):,} of {n_all:,}   median {med:.3f}  (IQR {q1:.3f} - {q3:.3f})")
print()

# Original sample (all current + never), same CV procedure for a like-for-like comparison
results = {"Original (self-report)": evaluate(df)}
print(line + "\nOriginal sample (self-report only)\n" + line)
print_eval(results["Original (self-report)"])
print()

# ---------------------------------------------------------------------------
# 2-5. Confirmed samples at 3 and 10 ng/mL
# ---------------------------------------------------------------------------
for cutoff in [3, 10]:
    never = df["smoking_status"] == "never"
    current = df["smoking_status"] == "current"
    hi = df["LBXCOT"] >= cutoff
    lo = df["LBXCOT"] < cutoff
    print(line + f"\nCutoff {cutoff} ng/mL\n" + line)
    n_never, n_cur = (never & has_cot).sum(), (current & has_cot).sum()
    print(f"  Never smokers with cotinine >= {cutoff}: {(never & hi).sum():,} of {n_never:,} "
          f"({(never & hi).sum() / n_never:.1%})  -> excluded")
    print(f"  Current smokers with cotinine <  {cutoff}: {(current & lo).sum():,} of {n_cur:,} "
          f"({(current & lo).sum() / n_cur:.1%})  -> excluded")
    print(f"  No cotinine: {(~has_cot).sum():,} -> excluded")
    confirmed = df[(never & lo) | (current & hi)]
    r = evaluate(confirmed)
    results[f"Confirmed ({cutoff} ng/mL)"] = r
    print_eval(r)
    print()

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print(line + "\nSummary: 5-fold CV AUC, current vs never smokers\n" + line)
print(f"{'Sample':<24}{'n':>7}{'%cur':>7}{'A demo':>9}{'C diet':>9}{'E best':>9}"
      f"{'Delta E-A [95% CI]':>27}")
for name, r in results.items():
    print(f"{name:<24}{r['n']:>7,}{r['n_current'] / r['n']:>7.1%}{r['cv_A']:>9.3f}{r['cv_C']:>9.3f}"
          f"{r['cv_E']:>9.3f}   {r['delta_EA']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]")
