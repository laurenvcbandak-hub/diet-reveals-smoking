"""Main analysis: does one day of diet predict current (vs never) smoking beyond demographics?

Reads outputs/features.csv; writes outputs/model_results.csv and outputs/best_model.joblib.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
N_BOOT = 1000
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = pd.read_csv("outputs/features.csv")
cohort_cols = pd.read_csv("outputs/cohort.csv", nrows=0).columns
diet_cols = [c for c in df.columns if c not in cohort_cols]
assert len(diet_cols) == 40, f"expected 40 diet features, found {len(diet_cols)}"

df = df[df["smoking_status"].isin(["current", "never"])]
df = df[df["kcal"].notna()]                      # the person with no food records
df["no_caffeine"] = df["first_caffeine_h"].isna().astype(int)
y = (df["smoking_status"] == "current").astype(int).to_numpy()

DEMO_NUM = ["RIDAGEYR"]
DEMO_CAT = ["RIAGENDR", "RIDRETH3", "DMDEDUC2"]
DEMO_IMPUTE = ["INDFMPIR"]
BODY = ["BMXBMI", "BMXWAIST"]
DIET = diet_cols + ["no_caffeine"]

X = df[DEMO_NUM + DEMO_CAT + DEMO_IMPUTE + BODY + DIET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

print(f"Current + never smokers: {len(df):,} ({y.mean():.1%} current)")
print(f"Train: {len(X_train):,}   Test: {len(X_test):,} (held out until the end)\n")


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def preprocessor(demo, body, diet, scale):
    num_tail = [StandardScaler()] if scale else []
    blocks = []
    if demo:
        blocks += [
            ("age", Pipeline([("pass", "passthrough")] + [("s", s) for s in num_tail]), DEMO_NUM),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value=-1)),   # missing = own category
                ("oh", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
            ]), DEMO_CAT),
            ("pir", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))]
                             + [("s", s) for s in num_tail]), DEMO_IMPUTE),
        ]
    if body:
        blocks.append(("body", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))]
                                        + [("s", s) for s in num_tail]), BODY))
    if diet:
        blocks.append(("diet", Pipeline([("imp", SimpleImputer(strategy="median"))]
                                        + [("s", s) for s in num_tail]), DIET))
    return ColumnTransformer(blocks)


def make_model(demo, body, diet, method):
    if method == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=5000)   # L2 is the default penalty
        return Pipeline([("prep", preprocessor(demo, body, diet, scale=True)), ("clf", clf)])
    clf = HistGradientBoostingClassifier(random_state=SEED)
    return Pipeline([("prep", preprocessor(demo, body, diet, scale=False)), ("clf", clf)])


specs = {
    "A  demo (logreg)":             (True,  False, False, "logreg"),
    "B  demo+body (logreg)":        (True,  True,  False, "logreg"),
    "C  diet (logreg)":             (False, False, True,  "logreg"),
    "C  diet (HGB)":                (False, False, True,  "hgb"),
    "D  demo+diet (logreg)":        (True,  False, True,  "logreg"),
    "D  demo+diet (HGB)":           (True,  False, True,  "hgb"),
    "E  demo+body+diet (logreg)":   (True,  True,  True,  "logreg"),
    "E  demo+body+diet (HGB)":      (True,  True,  True,  "hgb"),
}

# ---------------------------------------------------------------------------
# 5-fold CV on the training set (same folds for every model -> paired OOF predictions)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X_train, y_train))
oof, cv_rows = {}, []
for name, spec in specs.items():
    pred = np.zeros(len(y_train))
    fold_aucs = []
    for tr, va in folds:
        m = make_model(*spec).fit(X_train.iloc[tr], y_train[tr])
        pred[va] = m.predict_proba(X_train.iloc[va])[:, 1]
        fold_aucs.append(roc_auc_score(y_train[va], pred[va]))
    oof[name] = pred
    cv_rows.append({"model": name, "cv_auc_mean": np.mean(fold_aucs),
                    "cv_auc_sd": np.std(fold_aucs), "oof_auc": roc_auc_score(y_train, pred)})
cv_table = pd.DataFrame(cv_rows)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def boot_idx(y_true, n=N_BOOT):
    idx = []
    while len(idx) < n:
        i = rng.integers(0, len(y_true), len(y_true))
        if 0 < y_true[i].sum() < len(i):
            idx.append(i)
    return idx


def auc_ci(y_true, p):
    stats = [roc_auc_score(y_true[i], p[i]) for i in boot_idx(y_true)]
    return roc_auc_score(y_true, p), *np.percentile(stats, [2.5, 97.5])


def delta_ci(y_true, p_new, p_base):
    stats = [roc_auc_score(y_true[i], p_new[i]) - roc_auc_score(y_true[i], p_base[i])
             for i in boot_idx(y_true)]
    est = roc_auc_score(y_true, p_new) - roc_auc_score(y_true, p_base)
    return est, *np.percentile(stats, [2.5, 97.5])


# ---------------------------------------------------------------------------
# 1. Delta AUC, D vs A and E vs B (OOF predictions, paired bootstrap)
# ---------------------------------------------------------------------------
comparisons = [
    ("D  demo+diet (logreg)", "A  demo (logreg)"),
    ("D  demo+diet (HGB)", "A  demo (logreg)"),
    ("E  demo+body+diet (logreg)", "B  demo+body (logreg)"),
    ("E  demo+body+diet (HGB)", "B  demo+body (logreg)"),
]
delta_rows = []
for new, base in comparisons:
    est, lo, hi = delta_ci(y_train, oof[new], oof[base])
    delta_rows.append({"comparison": f"{new.strip()}  vs  {base[:1]}", "delta_auc": est,
                       "ci_low": lo, "ci_high": hi})
delta_table = pd.DataFrame(delta_rows)

# ---------------------------------------------------------------------------
# 2. Best of D/E by CV AUC -> fit on full train, evaluate once on test
# ---------------------------------------------------------------------------
de = cv_table[cv_table["model"].str[0].isin(["D", "E"])]
best_name = de.loc[de["cv_auc_mean"].idxmax(), "model"]
best = make_model(*specs[best_name]).fit(X_train, y_train)
base = make_model(*specs["A  demo (logreg)"]).fit(X_train, y_train)
p_best = best.predict_proba(X_test)[:, 1]
p_base = base.predict_proba(X_test)[:, 1]
test_best = auc_ci(y_test, p_best)
test_base = auc_ci(y_test, p_base)
test_delta = delta_ci(y_test, p_best, p_base)

# ---------------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------------
Path("outputs").mkdir(exist_ok=True)
rows = []
for r in cv_rows:
    rows.append({"stage": "cv_train", "model": r["model"], "metric": "auc_cv_mean",
                 "estimate": r["cv_auc_mean"], "sd": r["cv_auc_sd"]})
    rows.append({"stage": "cv_train", "model": r["model"], "metric": "auc_oof",
                 "estimate": r["oof_auc"]})
for r in delta_rows:
    rows.append({"stage": "cv_train", "model": r["comparison"], "metric": "delta_auc_oof",
                 "estimate": r["delta_auc"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
for name, (est, lo, hi) in [(best_name, test_best), ("A  demo (logreg)", test_base)]:
    rows.append({"stage": "test", "model": name, "metric": "auc",
                 "estimate": est, "ci_low": lo, "ci_high": hi})
rows.append({"stage": "test", "model": f"{best_name.strip()}  vs  A", "metric": "delta_auc",
             "estimate": test_delta[0], "ci_low": test_delta[1], "ci_high": test_delta[2]})
pd.DataFrame(rows).to_csv("outputs/model_results.csv", index=False)
joblib.dump({"model": best, "name": best_name, "features": list(X.columns)},
            "outputs/best_model.joblib")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
line = "=" * 74
print(line)
print("5-fold CV on training set (AUC)")
print(line)
print(f"{'Model':<30}{'CV mean':>10}{'CV SD':>8}{'OOF AUC':>10}")
for r in cv_rows:
    print(f"{r['model']:<30}{r['cv_auc_mean']:>10.3f}{r['cv_auc_sd']:>8.3f}{r['oof_auc']:>10.3f}")

print(f"\n{line}\nDelta AUC from adding diet (OOF, {N_BOOT} paired bootstraps, 95% CI)\n{line}")
for r in delta_rows:
    print(f"{r['comparison']:<45}{r['delta_auc']:>+8.3f}   [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")

print(f"\n{line}\nHeld-out test set (n={len(y_test):,}, evaluated once; 95% bootstrap CI)\n{line}")
print(f"Best D/E model by CV AUC: {best_name.strip()}")
print(f"{best_name.strip():<45}AUC {test_best[0]:.3f}   [{test_best[1]:.3f}, {test_best[2]:.3f}]")
print(f"{'A  demo (logreg)':<45}AUC {test_base[0]:.3f}   [{test_base[1]:.3f}, {test_base[2]:.3f}]")
print(f"{'Delta (best - A)':<45}    {test_delta[0]:+.3f}   [{test_delta[1]:+.3f}, {test_delta[2]:+.3f}]")
print("\nSaved outputs/model_results.csv and outputs/best_model.joblib")
