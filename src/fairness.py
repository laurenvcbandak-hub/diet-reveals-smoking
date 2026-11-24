"""Do the models work equally well across sex, age and ethnicity groups?

Out-of-fold predictions (5-fold, current vs never smokers) for models A, C and E (HGB),
subgroup AUCs with paired bootstrap CIs. Writes outputs/fairness.csv and
outputs/figures/fairness_auc.png.
"""
from pathlib import Path

import matplotlib
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SEED = 42
N_BOOT = 1000
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data (same preparation as train_models.py)
# ---------------------------------------------------------------------------
df = pd.read_csv("outputs/features.csv")
cohort_cols = pd.read_csv("outputs/cohort.csv", nrows=0).columns
diet_cols = [c for c in df.columns if c not in cohort_cols]
df = df[df["smoking_status"].isin(["current", "never"]) & df["kcal"].notna()].reset_index(drop=True)
df["no_caffeine"] = df["first_caffeine_h"].isna().astype(int)
y = (df["smoking_status"] == "current").astype(int).to_numpy()

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


MODELS = {
    "A: demographics (logreg)": (True, False, False, "logreg"),
    "C: diet only (logreg)": (False, False, True, "logreg"),
    "E: demo+body+diet (HGB)": (True, True, True, "hgb"),
}

# ---------------------------------------------------------------------------
# 1. Out-of-fold predictions on all current + never smokers
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = {name: np.zeros(len(y)) for name in MODELS}
for tr, va in cv.split(X, y):
    for name, spec in MODELS.items():
        m = make_model(*spec).fit(X.iloc[tr], y[tr])
        oof[name][va] = m.predict_proba(X.iloc[va])[:, 1]

# ---------------------------------------------------------------------------
# 2. Subgroups (P_DEMO codebook)
# RIAGENDR: 1 = male, 2 = female
# RIDRETH3: 1 = Mexican American, 2 = Other Hispanic, 3 = Non-Hispanic White,
#           4 = Non-Hispanic Black, 6 = Non-Hispanic Asian, 7 = Other race incl. multiracial
# ---------------------------------------------------------------------------
SEX = {1: "Male", 2: "Female"}
ETH = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
       4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other/multiracial"}
age_group = pd.cut(df["RIDAGEYR"], bins=[20, 40, 60, np.inf], right=False,
                   labels=["20-39", "40-59", "60+"])

subgroups = [("Overall", "All", np.ones(len(df), bool))]
subgroups += [("Sex", lab, (df["RIAGENDR"] == code).to_numpy()) for code, lab in SEX.items()]
subgroups += [("Age", lab, (age_group == lab).to_numpy()) for lab in age_group.cat.categories]
subgroups += [("Ethnicity", lab, (df["RIDRETH3"] == code).to_numpy()) for code, lab in ETH.items()]

rng = np.random.default_rng(SEED)
rows = []
for dim, label, mask in subgroups:
    yg = y[mask]
    preds = {name: p[mask] for name, p in oof.items()}
    # same bootstrap resamples for every model (paired), resampling until both classes present
    idx = []
    while len(idx) < N_BOOT:
        i = rng.integers(0, len(yg), len(yg))
        if 0 < yg[i].sum() < len(i):
            idx.append(i)
    for name, p in preds.items():
        boots = [roc_auc_score(yg[i], p[i]) for i in idx]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"dimension": dim, "subgroup": label, "n": int(mask.sum()),
                     "n_current": int(yg.sum()), "pct_current": 100 * yg.mean(),
                     "model": name, "auc": roc_auc_score(yg, p), "ci_low": lo, "ci_high": hi})
res = pd.DataFrame(rows)
Path("outputs").mkdir(exist_ok=True)
res.to_csv("outputs/fairness.csv", index=False)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
short = {"A: demographics (logreg)": "A demo", "C: diet only (logreg)": "C diet",
         "E: demo+body+diet (HGB)": "E best"}
wide = res.pivot_table(index=["dimension", "subgroup"], columns="model", sort=False,
                       values=["auc", "ci_low", "ci_high"])
meta = res.drop_duplicates(["dimension", "subgroup"]).set_index(["dimension", "subgroup"])

print(f"Out-of-fold predictions, 5-fold, current vs never smokers (n={len(y):,}); "
      f"95% CI from {N_BOOT} paired bootstraps\n")
hdr = f"{'Subgroup':<22}{'n':>6}{'Current':>13}" + "".join(f"{short[m]:>22}" for m in MODELS)
print(hdr)
print("-" * len(hdr))
last_dim = None
for (dim, sub), r in meta.iterrows():
    if dim != last_dim:
        print(f"{dim}")
        last_dim = dim
    cells = "".join(
        f"{wide.loc[(dim, sub), ('auc', m)]:>8.3f} [{wide.loc[(dim, sub), ('ci_low', m)]:.2f}-"
        f"{wide.loc[(dim, sub), ('ci_high', m)]:.2f}]" for m in MODELS)
    print(f"  {sub:<20}{r['n']:>6,}{r['n_current']:>7,} ({r['pct_current']:>4.1f}%)" + cells)
print("\nSaved outputs/fairness.csv")

# ---------------------------------------------------------------------------
# 3. Figure: AUC with CI per subgroup, one marker per model, overall AUC as vertical lines
# ---------------------------------------------------------------------------
COLORS = {"A: demographics (logreg)": "#2a78d6", "C: diet only (logreg)": "#eb6834",
          "E: demo+body+diet (HGB)": "#1baf7a"}
MARKERS = {"A: demographics (logreg)": "o", "C: diet only (logreg)": "s",
           "E: demo+body+diet (HGB)": "D"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

groups = [(d, s) for d, s, _ in subgroups if d != "Overall"]
ypos, ticks, y_at, dim_heads = [], [], 0.0, []
for d, s in groups:
    if not dim_heads or dim_heads[-1][0] != d:
        if dim_heads:
            y_at += 0.6
        dim_heads.append((d, y_at))
    ypos.append(y_at)
    ticks.append(f"{s}  (n={meta.loc[(d, s), 'n']:,})")
    y_at += 1
ypos = np.array(ypos)

fig, ax = plt.subplots(figsize=(8.5, 7.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
offsets = dict(zip(MODELS, [-0.22, 0, 0.22]))
for name in MODELS:
    overall = res[(res["dimension"] == "Overall") & (res["model"] == name)]["auc"].iloc[0]
    ax.axvline(overall, color=COLORS[name], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    sub = res[res["model"] == name].set_index(["dimension", "subgroup"]).loc[groups]
    yy = ypos + offsets[name]
    ax.hlines(yy, sub["ci_low"], sub["ci_high"], color=COLORS[name], linewidth=2, zorder=2)
    ax.scatter(sub["auc"], yy, color=COLORS[name], marker=MARKERS[name], s=42, zorder=3,
               edgecolors=SURFACE, linewidths=1.5)
ax.set_yticks(ypos, ticks)
ax.invert_yaxis()
for d, yh in dim_heads:
    ax.text(-0.01, yh - 0.55, d.upper(), transform=ax.get_yaxis_transform(), ha="right",
            fontsize=8.5, fontweight="bold", color=INK2)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, length=0, labelsize=9.5)
ax.set_xlabel("AUC, out-of-fold (95% bootstrap CI); dashed lines = each model's overall AUC",
              color=INK2, fontsize=9.5)
handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linewidth=2, markersize=7,
                  label=f"{m}  (overall {res[(res['dimension'] == 'Overall') & (res['model'] == m)]['auc'].iloc[0]:.3f})")
           for m in MODELS]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, frameon=False,
          fontsize=9, labelcolor=INK2)
fig.suptitle("Model performance by subgroup", x=0.02, ha="left", fontsize=13,
             fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(FIG_DIR / "fairness_auc.png", dpi=200, facecolor=SURFACE)
print(f"Saved {FIG_DIR / 'fairness_auc.png'}")
