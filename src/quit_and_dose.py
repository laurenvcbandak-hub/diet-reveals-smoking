"""Secondary analyses: does the diet smoker score track time since quitting and cigarettes per day?

Diet smoker score = predicted probability of current smoking from the diet-only logistic
regression (model C in train_models.py). Reads outputs/features.csv; writes two figures
to outputs/figures/.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.anova import anova_lm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = pd.read_csv("outputs/features.csv")
cohort_cols = pd.read_csv("outputs/cohort.csv", nrows=0).columns
diet_cols = [c for c in df.columns if c not in cohort_cols]
df = df[df["kcal"].notna()].copy()                       # drop the person with no food records
df["no_caffeine"] = df["first_caffeine_h"].isna().astype(int)
DIET = diet_cols + ["no_caffeine"]


def diet_model():
    """Model C: diet-only L2 logistic regression, same pipeline as train_models.py."""
    return Pipeline([
        ("prep", ColumnTransformer([("diet", Pipeline([
            ("imp", SimpleImputer(strategy="median")), ("s", StandardScaler())]), DIET)])),
        ("clf", LogisticRegression(C=1.0, max_iter=5000)),
    ])


# ---------------------------------------------------------------------------
# 1. Diet smoker score
# ---------------------------------------------------------------------------
cn = df[df["smoking_status"].isin(["current", "never"])]
y_cn = (cn["smoking_status"] == "current").astype(int)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
df.loc[cn.index, "score"] = cross_val_predict(diet_model(), cn[DIET], y_cn, cv=cv,
                                              method="predict_proba")[:, 1]
full = diet_model().fit(cn[DIET], y_cn)
former = df["smoking_status"] == "former"
df.loc[former, "score"] = full.predict_proba(df.loc[former, DIET])[:, 1]

print("Diet smoker score = P(current smoker | diet), model C (diet-only logistic regression)")
print(f"  current + never: 5-fold out-of-fold predictions (n={len(cn):,})")
print(f"  former: model fit on all current + never smokers (n={former.sum():,})\n")

# ---------------------------------------------------------------------------
# 2. Time since quitting
# P_SMQ codebook: SMQ050U 1=days, 2=weeks, 3=months, 4=years, 7=refused, 9=don't know.
# SMQ050Q 66666 = "50 or more years" (no unit recorded; build_cohort.py top-coded it to 50),
#         77777 = refused, 99999 = don't know (already missing in cohort.csv).
# ---------------------------------------------------------------------------
per_year = {1: 365.25, 2: 52.1775, 3: 12.0, 4: 1.0}
q, u = df["SMQ050Q"], df["SMQ050U"]
df["quit_years"] = q / u.map(per_year)
df.loc[(q == 50) & u.isna(), "quit_years"] = 50.0         # the "50 or more years" answers
df.loc[~former, "quit_years"] = np.nan

n_former = former.sum()
n_quit = df.loc[former, "quit_years"].notna().sum()
print(f"Former smokers with usable quit time: {n_quit:,} of {n_former:,} "
      f"(missing {n_former - n_quit}: refused / don't know / blank)\n")

bins = [0, 1, 5, 10, 20, np.inf]
labels = ["<1 y", "1-5 y", "5-10 y", "10-20 y", ">20 y"]
df["quit_group"] = pd.cut(df["quit_years"], bins=bins, labels=labels, right=False)


def mean_ci(s):
    s = s.dropna()
    m, se = s.mean(), s.std(ddof=1) / np.sqrt(len(s))
    h = stats.t.ppf(0.975, len(s) - 1) * se
    return len(s), m, m - h, m + h


def summary_table(groups):
    rows = [(name, *mean_ci(s)) for name, s in groups]
    return pd.DataFrame(rows, columns=["group", "n", "mean", "ci_low", "ci_high"])


never_s = df.loc[df["smoking_status"] == "never", "score"]
current_s = df.loc[df["smoking_status"] == "current", "score"]
quit_groups = [("never", never_s)]
quit_groups += [(f"quit {g}", df.loc[df["quit_group"] == g, "score"]) for g in labels[::-1]]
quit_groups += [("current", current_s)]
quit_tab = summary_table(quit_groups)

line = "=" * 60
print(line + "\nDiet smoker score by time since quitting (mean, 95% CI)\n" + line)
print(f"{'Group':<16}{'n':>7}{'Mean':>9}{'95% CI':>20}")
for r in quit_tab.itertuples():
    print(f"{r.group:<16}{r.n:>7,}{r.mean:>9.3f}      [{r.ci_low:.3f}, {r.ci_high:.3f}]")

# Covariates: education missing as its own category, income median-imputed + flag
df["educ"] = df["DMDEDUC2"].fillna(0).astype(int).astype(str)
df["pir_missing"] = df["INDFMPIR"].isna().astype(int)
df["pir"] = df["INDFMPIR"].fillna(df["INDFMPIR"].median())
df["sex"] = df["RIAGENDR"].astype(int)
ADJ = "RIDAGEYR + C(sex) + C(educ) + pir + pir_missing"

fq = df[former & df["quit_years"].notna()]
# constraints="center" keeps the spline from duplicating the intercept (full 4 df)
spline = smf.ols(f"score ~ cr(quit_years, df=4, constraints='center') + {ADJ}", data=fq).fit()
base = smf.ols(f"score ~ {ADJ}", data=fq).fit()
ftest = anova_lm(base, spline)
print(f"\nFormer smokers (n={len(fq):,}): score ~ natural spline(years since quitting, 4 df)")
print("  adjusted for age, sex, education, income")
print(f"  Quit-time effect: F({int(ftest['df_diff'][1])}, {int(ftest['df_resid'][1])}) = "
      f"{ftest['F'][1]:.2f}, p = {ftest['Pr(>F)'][1]:.2g}")
lin = smf.ols(f"score ~ quit_years + {ADJ}", data=fq).fit()
print(f"  (Linear trend for reference: {lin.params['quit_years'] * 10:+.4f} per 10 years, "
      f"p = {lin.pvalues['quit_years']:.2g})\n")

# ---------------------------------------------------------------------------
# 3. Dose: cigarettes per day among current smokers
# ---------------------------------------------------------------------------
cur = df[(df["smoking_status"] == "current")].copy()
cur["cpd_group"] = pd.cut(cur["SMD650"], bins=[0, 5, 10, 20, np.inf],
                          labels=["1-5", "6-10", "11-20", ">20"])
dose_tab = summary_table([(f"{g} cig/day", cur.loc[cur["cpd_group"] == g, "score"])
                          for g in ["1-5", "6-10", "11-20", ">20"]])
print(line + "\nDiet smoker score by cigarettes/day, current smokers (mean, 95% CI)\n" + line)
print(f"{'Group':<16}{'n':>7}{'Mean':>9}{'95% CI':>20}")
for r in dose_tab.itertuples():
    print(f"{r.group:<16}{r.n:>7,}{r.mean:>9.3f}      [{r.ci_low:.3f}, {r.ci_high:.3f}]")
print(f"(missing SMD650: {cur['SMD650'].isna().sum()})")

cd = cur[cur["SMD650"].notna()]
dose = smf.ols(f"score ~ SMD650 + {ADJ}", data=cd).fit()
b, ci = dose.params["SMD650"], dose.conf_int().loc["SMD650"]
print(f"\nCurrent smokers (n={len(cd):,}): score ~ cigarettes/day, adjusted for age, sex, "
      "education, income")
print(f"  Coefficient: {b:+.4f} per cigarette/day  ({b * 10:+.3f} per 10 cig/day), "
      f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], p = {dose.pvalues['SMD650']:.2g}\n")

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
BLUE, INK, INK2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def dot_plot(tab, title, subtitle, xlabel, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(tab))
    for ref, name in [(never_s.mean(), "Never smokers"), (current_s.mean(), "Current smokers")]:
        ax.axhline(ref, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(len(tab) - 0.55, ref, f"{name} {ref:.3f}", va="bottom", ha="right",
                fontsize=9, color=INK2)
    ax.errorbar(x, tab["mean"], yerr=[tab["mean"] - tab["ci_low"], tab["ci_high"] - tab["mean"]],
                fmt="o", color=BLUE, markersize=8, elinewidth=2, capsize=0, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xticks(x, [f"{g}\nn={n:,}" for g, n in zip(tab["group"], tab["n"])])
    ax.set_xlim(-0.5, len(tab) - 0.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9.5)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_ylabel("Diet smoker score (mean, 95% CI)", color=INK2, fontsize=9.5)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK2, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {path}")


qt = quit_tab[quit_tab["group"].str.startswith("quit")].copy()
qt["group"] = qt["group"].str.replace("quit ", "", regex=False)
dot_plot(qt, "Diet smoker score by time since quitting",
         "Former smokers; dashed lines = never and current smoker means",
         "Time since quitting", FIG_DIR / "score_by_quit_time.png")
dt = dose_tab.copy()
dt["group"] = dt["group"].str.replace(" cig/day", "", regex=False)
dot_plot(dt, "Diet smoker score by cigarettes per day",
         "Current smokers; dashed lines = never and current smoker means",
         "Cigarettes per day", FIG_DIR / "score_by_cigs_per_day.png")
