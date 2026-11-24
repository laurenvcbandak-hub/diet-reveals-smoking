"""Explain the best model: which features drive the predictions of current smoking?

Reads outputs/best_model.joblib and outputs/features.csv.
Writes outputs/figures/feature_importance.png (and shap_summary.png if shap is installed).
"""
import importlib.util
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Model and the same train/test split as train_models.py
# ---------------------------------------------------------------------------
saved = joblib.load("outputs/best_model.joblib")
model, model_name, feature_order = saved["model"], saved["name"].strip(), saved["features"]

df = pd.read_csv("outputs/features.csv")
cohort_cols = pd.read_csv("outputs/cohort.csv", nrows=0).columns
diet_cols = [c for c in df.columns if c not in cohort_cols]
df = df[df["smoking_status"].isin(["current", "never"])]
df = df[df["kcal"].notna()]
df["no_caffeine"] = df["first_caffeine_h"].isna().astype(int)
y = (df["smoking_status"] == "current").astype(int).to_numpy()
X = df[feature_order]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Model: {model_name}")
print(f"Test set: n={len(y_test):,}, AUC {test_auc:.3f} (should match train_models.py)\n")

# ---------------------------------------------------------------------------
# 2. Permutation importance on the test set (permutes each raw input column)
# ---------------------------------------------------------------------------
perm = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                              n_repeats=30, random_state=SEED, n_jobs=-1)
imp = pd.DataFrame({"feature": X_test.columns, "mean": perm.importances_mean,
                    "sd": perm.importances_std}).sort_values("mean", ascending=False)
top = imp.head(15)
print("=" * 62)
print("Permutation importance, test set (drop in AUC, 30 repeats)")
print("=" * 62)
print(f"{'Feature':<26}{'Mean':>10}{'SD':>10}")
for r in top.itertuples():
    print(f"{r.feature:<26}{r.mean:>10.4f}{r.sd:>10.4f}")
print()

# ---------------------------------------------------------------------------
# 3. SHAP summary (optional)
# ---------------------------------------------------------------------------
if importlib.util.find_spec("shap") is None:
    print("SHAP: package 'shap' is not installed - skipping the SHAP summary plot.\n")
else:
    import shap
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = pd.DataFrame(prep.transform(X_test), columns=prep.get_feature_names_out())
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    shap.summary_plot(sv, Xt, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=200)
    plt.close()
    print(f"SHAP: saved {FIG_DIR / 'shap_summary.png'}\n")

# ---------------------------------------------------------------------------
# 4. Diet-only logistic regression (model C): standardized coefficients
# ---------------------------------------------------------------------------
DIET = diet_cols + ["no_caffeine"]
diet_lr = Pipeline([
    ("prep", ColumnTransformer([("diet", Pipeline([
        ("imp", SimpleImputer(strategy="median")), ("s", StandardScaler())]), DIET)])),
    ("clf", LogisticRegression(C=1.0, max_iter=5000)),
]).fit(X_train, y_train)
coef = pd.Series(diet_lr.named_steps["clf"].coef_[0], index=DIET)
coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(15)
print("=" * 62)
print("Diet-only logistic regression (C), training set")
print("Largest standardized coefficients (log-odds per 1 SD; + = more likely smoker)")
print("=" * 62)
print(f"{'Feature':<26}{'Coef':>10}{'Odds ratio':>12}   Direction")
for f, b in coef.items():
    print(f"{f:<26}{b:>+10.3f}{np.exp(b):>12.2f}   {'+ smoker' if b > 0 else '- smoker'}")
print()

# ---------------------------------------------------------------------------
# 5. Bar chart of permutation importance
# ---------------------------------------------------------------------------
BLUE, INK, INK2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plot = top.iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.barh(plot["feature"], plot["mean"], height=0.6, color=BLUE, zorder=2)
ax.errorbar(plot["mean"], plot["feature"], xerr=plot["sd"], fmt="none",
            ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
ax.axvline(0, color=INK2, linewidth=1, zorder=1)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, length=0, labelsize=9.5)
ax.set_xlabel("Drop in test-set AUC when the feature is shuffled (mean ± SD, 30 repeats)",
              color=INK2, fontsize=9.5)
fig.suptitle("What drives the smoking prediction?", x=0.02, ha="left",
             fontsize=13, fontweight="bold", color=INK)
ax.set_title(f"Permutation importance, top 15 features - {model_name}, test AUC {test_auc:.3f}",
             loc="left", fontsize=9.5, color=INK2, pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance.png", dpi=200, facecolor=SURFACE)
print(f"Saved {FIG_DIR / 'feature_importance.png'}")
