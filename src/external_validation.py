"""External validation on NHANES 2015-2016 (people never used for training).

Builds the 2015-2016 cohort and features with the same functions as build_cohort.py and
build_features.py, then evaluates model E (outputs/best_model.joblib, NOT refit) and baseline A
(refit on the 2017-2020 training set exactly as in train_models.py).

Writes outputs/cohort_2015.csv, outputs/features_2015.csv, outputs/external_validation.csv and
outputs/figures/calibration_external.png.
"""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_cohort import build_cohort, print_funnel
from build_features import build_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
N_BOOT = 1000
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
line = "=" * 72

COHORT_2015 = {"demo": "data/2015/DEMO_I.xpt", "tot": "data/2015/DR1TOT_I.xpt",
               "smq": "data/2015/SMQ_I.xpt", "bmx": "data/2015/BMX_I.xpt", "cot": None,
               "weight": "WTDRD1"}
FEATURES_2015 = {"iff": "data/2015/DR1IFF_I.xpt", "fcd": "data/2015/DRXFCD_I.xpt",
                 "tot": "data/2015/DR1TOT_I.xpt", "weight": "WTDRD1"}

# ---------------------------------------------------------------------------
# 2 (checked first). Do all needed columns exist in both cycles?
# ---------------------------------------------------------------------------
NEEDED = {
    "DEMO": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR", "SDMVPSU", "SDMVSTRA"],
    "DR1TOT": ["SEQN", "DR1DRSTZ", "<weight>", "DR1DAY", "DR1_320Z"],
    "SMQ": ["SEQN", "SMQ020", "SMQ040", "SMD650", "SMQ050Q", "SMQ050U"],
    "BMX": ["SEQN", "BMXBMI", "BMXWAIST"],
    "DR1IFF": ["SEQN", "DR1_020", "DR1IFDCD", "DR1IKCAL", "DR1IPROT", "DR1ICARB", "DR1ISUGR",
               "DR1IFIBE", "DR1ITFAT", "DR1IALCO", "DR1ICAFF"],
    "DRXFCD": ["DRXFDCD", "DRXFCSD", "DRXFCLD"],
}
PATHS = {"2017-2020": lambda f: f"data/P_{f}.xpt", "2015-2016": lambda f: f"data/2015/{f}_I.xpt"}
WEIGHT = {"2017-2020": "WTDRD1PP", "2015-2016": "WTDRD1"}


def columns_of(path):
    with pd.read_sas(path, format="xport", encoding="latin-1", chunksize=1) as r:
        return set(next(r).columns)


print(line + "\nColumn check: 2017-2020 vs 2015-2016\n" + line)
problems = []
for f, cols in NEEDED.items():
    for cycle, path in PATHS.items():
        have = columns_of(path(f))
        need = [WEIGHT[cycle] if c == "<weight>" else c for c in cols]
        missing = [c for c in need if c not in have]
        if missing:
            problems.append(f"{cycle} {f}: missing {missing}")
print(f"All {sum(len(c) for c in NEEDED.values())} needed columns present in both cycles"
      if not problems else "\n".join(problems))
print("Differences handled:")
print("  - Day-1 dietary weight: WTDRD1PP (2017-2020) vs WTDRD1 (2015-2016); not a model input")
print("  - No cotinine file for 2015-2016: LBXCOT is left missing (not a model input)")
print("  - Special codes (DMDEDUC2 7/9, SMD650 777/999, SMQ050Q 66666/77777/99999) are the same")
if problems:
    raise SystemExit("Missing columns - stopping.")
print()

# ---------------------------------------------------------------------------
# 1. 2015-2016 cohort and features, same functions as 2017-2020
# ---------------------------------------------------------------------------
print(line + "\nNHANES 2015-2016 cohort\n" + line)
cohort15, funnel = build_cohort(COHORT_2015)
print_funnel(funnel)
for s in ["current", "former", "never"]:
    print(f"  {s:<8} {(cohort15['smoking_status'] == s).sum():>6,}")
feat15, feature_cols, n_unmatched = build_features(cohort15, FEATURES_2015, verbose=False)
print(f"\nUnmatched food codes (DR1IFDCD not in DRXFCD_I): {n_unmatched}")
print(f"Cohort members with no food records: {feat15['kcal'].isna().sum()}")
cohort15.to_csv("outputs/cohort_2015.csv", index=False)
feat15.to_csv("outputs/features_2015.csv", index=False)
print("Saved outputs/cohort_2015.csv and outputs/features_2015.csv\n")


def prepare(df):
    df = df[df["smoking_status"].isin(["current", "never"]) & df["kcal"].notna()].copy()
    df["no_caffeine"] = df["first_caffeine_h"].isna().astype(int)
    return df, (df["smoking_status"] == "current").astype(int).to_numpy()


# ---------------------------------------------------------------------------
# 3. Models: E loaded as-is; A refit on the 2017-2020 training set (train_models.py rules)
# ---------------------------------------------------------------------------
saved = joblib.load("outputs/best_model.joblib")
model_E, features = saved["model"], saved["features"]

df17, y17 = prepare(pd.read_csv("outputs/features.csv"))
X_train, X_test, y_train, y_test = train_test_split(
    df17[features], y17, test_size=0.2, stratify=y17, random_state=SEED)

model_A = Pipeline([
    ("prep", ColumnTransformer([
        ("age", Pipeline([("pass", "passthrough"), ("s", StandardScaler())]), ["RIDAGEYR"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                          ("oh", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))]),
         ["RIAGENDR", "RIDRETH3", "DMDEDUC2"]),
        ("pir", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("s", StandardScaler())]), ["INDFMPIR"]),
    ])),
    ("clf", LogisticRegression(C=1.0, max_iter=5000)),
]).fit(X_train, y_train)

df15, y15 = prepare(feat15)
X15 = df15[features]
missing_feats = [c for c in features if c not in df15.columns]
assert not missing_feats, f"features missing in 2015-2016: {missing_feats}"

# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)


def evaluate(X, y):
    pA, pE = model_A.predict_proba(X)[:, 1], model_E.predict_proba(X)[:, 1]
    boots = {"A": [], "E": [], "d": []}
    while len(boots["d"]) < N_BOOT:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            a, e = roc_auc_score(y[i], pA[i]), roc_auc_score(y[i], pE[i])
            boots["A"].append(a), boots["E"].append(e), boots["d"].append(e - a)
    aA, aE = roc_auc_score(y, pA), roc_auc_score(y, pE)
    r = {"n": len(y), "n_current": int(y.sum()),
         "auc_A": aA, "auc_E": aE, "delta": aE - aA,
         "brier_A": brier_score_loss(y, pA), "brier_E": brier_score_loss(y, pE),
         "brier_null": brier_score_loss(y, np.full(len(y), y.mean())),
         "mean_pred_A": pA.mean(), "mean_pred_E": pE.mean()}
    for k, key in [("A", "auc_A"), ("E", "auc_E"), ("d", "delta")]:
        r[f"{key}_lo"], r[f"{key}_hi"] = np.percentile(boots[k], [2.5, 97.5])
    for k, p in [("A", pA), ("E", pE)]:
        lp = np.log(p / (1 - p))
        r[f"cal_slope_{k}"] = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
        r[f"citl_{k}"] = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                                offset=lp).fit().params[0]
    return r, pA, pE


res_test, _, _ = evaluate(X_test, y_test)
res_ext, pA15, pE15 = evaluate(X15, y15)

print(line + "\nExternal validation: NHANES 2015-2016, current vs never smokers\n" + line)
print(f"n = {res_ext['n']:,} ({res_ext['n_current']:,} current, "
      f"{res_ext['n_current'] / res_ext['n']:.1%})   models trained on 2017-2020, not refit\n")
print(f"{'':<30}{'AUC [95% CI]':>24}{'Brier':>9}{'Cal. slope':>12}{'CITL':>8}")
for k, lab in [("A", "A demographics (logreg)"), ("E", "E demo+body+diet (HGB)")]:
    print(f"{lab:<30}{res_ext[f'auc_{k}']:>10.3f} [{res_ext[f'auc_{k}_lo']:.3f}, "
          f"{res_ext[f'auc_{k}_hi']:.3f}]{res_ext[f'brier_{k}']:>9.4f}"
          f"{res_ext[f'cal_slope_{k}']:>12.2f}{res_ext[f'citl_{k}']:>+8.2f}")
print(f"{'Delta AUC (E - A), paired':<30}{res_ext['delta']:>+10.3f} "
      f"[{res_ext['delta_lo']:+.3f}, {res_ext['delta_hi']:+.3f}]")
print(f"\nBrier of a no-skill model (always predicts prevalence): {res_ext['brier_null']:.4f}")
print(f"Observed % current: {res_ext['n_current'] / res_ext['n']:.1%}   mean predicted: "
      f"A {res_ext['mean_pred_A']:.1%}, E {res_ext['mean_pred_E']:.1%}")
print("Cal. slope 1 = ideal (<1 = predictions too extreme); "
      "CITL = calibration-in-the-large, 0 = ideal (log-odds)\n")

# ---------------------------------------------------------------------------
# 5. Summary: 2017-2020 test set vs 2015-2016 external
# ---------------------------------------------------------------------------
print(line + "\nSummary\n" + line)
print(f"{'Sample':<22}{'n':>7}{'%cur':>7}{'AUC A':>8}{'AUC E':>8}{'Delta E-A [95% CI]':>26}"
      f"{'Brier A':>9}{'Brier E':>9}")
summary = []
for name, r in [("2017-2020 test set", res_test), ("2015-2016 external", res_ext)]:
    print(f"{name:<22}{r['n']:>7,}{r['n_current'] / r['n']:>7.1%}{r['auc_A']:>8.3f}{r['auc_E']:>8.3f}"
          f"   {r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
          f"{r['brier_A']:>9.4f}{r['brier_E']:>9.4f}")
    summary.append({"sample": name, **r})
pd.DataFrame(summary).to_csv("outputs/external_validation.csv", index=False)
print("\nSaved outputs/external_validation.csv")

# ---------------------------------------------------------------------------
# Calibration plot (deciles of predicted risk)
# ---------------------------------------------------------------------------
BLUE, AQUA = "#2a78d6", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

fig, ax = plt.subplots(figsize=(6.4, 6.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(0.97, 0.93, "Perfect calibration", rotation=45, ha="right", va="top",
        fontsize=8.5, color=INK2, transform=ax.transAxes, rotation_mode="anchor")
for p, color, marker, lab in [(pA15, BLUE, "o", "A demographics"),
                              (pE15, AQUA, "D", "E demo+body+diet")]:
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y15, "b": bins}).groupby("b")
    pred, obs, n = g["p"].mean(), g["y"].mean(), g.size()
    se = np.sqrt(obs * (1 - obs) / n)
    ax.errorbar(pred, obs, yerr=1.96 * se, color=color, marker=marker, markersize=7,
                linewidth=2, elinewidth=1.2, capsize=0, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    ax.annotate(lab, (pred.iloc[-1], obs.iloc[-1]), xytext=(-8, 6), textcoords="offset points",
                ha="right", fontsize=9, color=INK2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.grid(color=GRID, linewidth=0.8, zorder=0)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=INK2, length=0, labelsize=9.5)
ax.set_xlabel("Mean predicted probability of current smoking (decile)", color=INK2, fontsize=9.5)
ax.set_ylabel("Observed proportion current smokers (95% CI)", color=INK2, fontsize=9.5)
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
fig.suptitle("Calibration on NHANES 2015-2016", x=0.03, ha="left", fontsize=13,
             fontweight="bold", color=INK)
ax.set_title(f"External validation, n={res_ext['n']:,}; models trained on 2017-2020, not refit",
             loc="left", fontsize=9, color=INK2, pad=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "calibration_external.png", dpi=200, facecolor=SURFACE)
print(f"Saved {FIG_DIR / 'calibration_external.png'}")
