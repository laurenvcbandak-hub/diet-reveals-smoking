"""
EDA: time between last eating occasion and bedtime vs HbA1c in non-diabetic adults.
NHANES 2017-March 2020 pre-pandemic (P_ files).

Run from project root:  python src/eda.py
Outputs: outputs/figures/*.png, outputs/tables/*.csv, outputs/eda_log.txt
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
DATA = Path("data")
OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "eda_log.txt", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def read(name):
    df = pd.read_sas(DATA / f"{name}.xpt", encoding="infer")
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-70, 0.0)  # SAS tiny-float zeros
    return df


def wmean(x, w):
    m = x.notna() & w.notna()
    return np.average(x[m], weights=w[m]) if m.any() else np.nan


# --------------------------------------------------------------------------
# 1. Load + inventory
# --------------------------------------------------------------------------
files = ["P_DEMO", "P_BMX", "P_DIQ", "P_GHB", "P_SLQ", "P_DR1TOT", "P_DR1IFF"]
raw = {f: read(f) for f in files}
log("=== 1. FILE INVENTORY ===")
for f, df in raw.items():
    log(f"{f:9s} rows={len(df):7d} cols={df.shape[1]:4d} unique SEQN={df.SEQN.nunique()}")

demo, bmx, diq, ghb, slq, tot, iff = (raw[f] for f in files)

# --------------------------------------------------------------------------
# 2. Build exposure: last eating occasion & bedtime
# --------------------------------------------------------------------------
MAIN_MEALS = {1, 2, 3, 4, 5, 10, 11, 12, 14}  # breakfast..cena (excl. snacks/drinks)
iff = iff[iff.DR1DRSTZ == 1].copy()
iff["t_h"] = iff.DR1_020 / 3600.0
iff = iff[iff.DR1_030Z != 9]  # drop "extended consumption" (sipping all day)
# eating occasions before 04:00 belong to the night *before* the recall day's bedtime
day = iff[iff.t_h >= 4]
occ = day.groupby(["SEQN", "t_h", "DR1_030Z"], as_index=False).DR1IKCAL.sum()

last_any = occ[occ.DR1IKCAL > 0].groupby("SEQN").t_h.max().rename("last_any")
last_50 = occ[occ.DR1IKCAL >= 50].groupby("SEQN").t_h.max().rename("last_50")
last_meal = occ[occ.DR1_030Z.isin(MAIN_MEALS)].groupby("SEQN").t_h.max().rename("last_meal")
kcal_after20 = (day[day.t_h >= 20].groupby("SEQN").DR1IKCAL.sum()
                .rename("kcal_after20"))


def hhmm(s):
    try:
        h, m = str(s).split(":")
        return int(h) + int(m) / 60
    except Exception:
        return np.nan


slq["bed_wd"] = slq.SLQ300.map(hhmm)
slq["bed_we"] = slq.SLQ320.map(hhmm)
log("\nNon-HH:MM bedtime strings (weekday):",
    slq.loc[slq.bed_wd.isna(), "SLQ300"].value_counts().to_dict())

# --------------------------------------------------------------------------
# 3. Merge + cohort funnel
# --------------------------------------------------------------------------
df = (demo.merge(ghb, on="SEQN", how="left")
          .merge(diq[["SEQN", "DIQ010", "DIQ160", "DIQ050", "DIQ070"]], on="SEQN", how="left")
          .merge(slq[["SEQN", "bed_wd", "bed_we", "SLD012", "SLD013", "SLQ050", "SLQ120"]],
                 on="SEQN", how="left")
          .merge(bmx[["SEQN", "BMXBMI", "BMXWAIST"]], on="SEQN", how="left")
          .merge(tot[["SEQN", "DR1DRSTZ", "DR1DAY", "DR1TKCAL", "DR1TCARB", "DR1TSUGR",
                      "DR1TFIBE", "DR1TALCO", "WTDRD1PP"]], on="SEQN", how="left"))
for s in [last_any, last_50, last_meal, kcal_after20]:
    df = df.merge(s, left_on="SEQN", right_index=True, how="left")
df["kcal_after20"] = df.kcal_after20.fillna(0)

# recall day Sun(1)-Thu(5): next day is a workday -> weekday bedtime; Fri/Sat -> weekend
df["bed"] = np.where(df.DR1DAY.between(1, 5), df.bed_wd, df.bed_we)
df["bed_c"] = np.where(df.bed < 12, df.bed + 24, df.bed)  # 01:00 -> 25
df["sleep_h"] = np.where(df.DR1DAY.between(1, 5), df.SLD012, df.SLD013)

funnel = []
def step(label, mask):
    global df
    df = df[mask]
    funnel.append((label, len(df)))

funnel.append(("All P_DEMO participants", len(df)))
step("Age >= 20", df.RIDAGEYR >= 20)
step("Not pregnant", df.RIDEXPRG != 1)
step("HbA1c measured", df.LBXGH.notna())
step("No diagnosed diabetes (DIQ010 != 1)", df.DIQ010 != 1)
step("No insulin / diabetes pills", (df.DIQ050 != 1) & (df.DIQ070 != 1))
step("HbA1c < 6.5% (excl. undiagnosed diabetes)", df.LBXGH < 6.5)
step("Reliable day-1 dietary recall", df.DR1DRSTZ == 1)
step("Bedtime reported", df.bed.notna())
step("Has an eating occasion >= 50 kcal", df.last_50.notna())
step("Bedtime 18:00-06:00 (excl. likely shift workers)", df.bed_c.between(18, 30))
df["gap"] = df.bed_c - df.last_50
df["gap_meal"] = df.bed_c - df.last_meal
df["gap_any"] = df.bed_c - df.last_any
step("Plausible gap (-1h to 12h)", df.gap.between(-1, 12))
df["gap"] = df.gap.clip(lower=0)

log("\n=== 2. COHORT FUNNEL ===")
for lbl, n in funnel:
    log(f"{n:7d}  {lbl}")
pd.DataFrame(funnel, columns=["step", "n"]).to_csv(TAB / "cohort_funnel.csv", index=False)

df["healthy"] = (df.LBXGH < 5.7).astype(int)
df["w"] = df.WTDRD1PP
df["female"] = (df.RIAGENDR == 2).astype(int)
df["race"] = df.RIDRETH3.map({1: "MexAm", 2: "OthHisp", 3: "NHWhite", 4: "NHBlack",
                               6: "NHAsian", 7: "Other"})
df["educ"] = df.DMDEDUC2.where(df.DMDEDUC2.between(1, 5))
df["cluster"] = df.SDMVSTRA.astype(int).astype(str) + "_" + df.SDMVPSU.astype(int).astype(str)
df.to_csv(OUT / "analytic_cohort.csv", index=False)

# --------------------------------------------------------------------------
# 4. Descriptives + missingness
# --------------------------------------------------------------------------
log("\n=== 3. DESCRIPTIVES (analytic cohort, survey-weighted where noted) ===")
log(f"N = {len(df)};  weighted % HbA1c<5.7: {100*wmean(df.healthy, df.w):.1f}%  "
    f"(unweighted {100*df.healthy.mean():.1f}%)")
desc_vars = ["RIDAGEYR", "BMXBMI", "BMXWAIST", "INDFMPIR", "LBXGH", "bed_c", "last_50",
             "gap", "gap_meal", "sleep_h", "DR1TKCAL", "kcal_after20"]
d = df[desc_vars].describe(percentiles=[.05, .25, .5, .75, .95]).T
d["missing_%"] = 100 * df[desc_vars].isna().mean()
d["wmean"] = [wmean(df[v], df.w) for v in desc_vars]
log(d.round(2).to_string())
d.to_csv(TAB / "descriptives.csv")

# --------------------------------------------------------------------------
# 5. Figures: distributions
# --------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].hist(df.LBXGH, bins=np.arange(3.8, 6.6, 0.1), color="#4C72B0", edgecolor="white")
ax[0, 0].axvline(5.7, color="crimson", ls="--", label="5.7% cutoff")
ax[0, 0].set(title="HbA1c (%)", xlabel="HbA1c"); ax[0, 0].legend()
ax[0, 1].hist(df.bed_c, bins=np.arange(18, 30.25, 0.25), color="#55A868", edgecolor="white")
ax[0, 1].set(title="Bedtime on recall night", xlabel="Hour (24 = midnight)")
ax[1, 0].hist(df.last_50, bins=np.arange(4, 24.25, 0.25), color="#C44E52", edgecolor="white")
ax[1, 0].set(title="Last eating occasion (>=50 kcal)", xlabel="Clock hour")
ax[1, 1].hist(df.gap, bins=np.arange(0, 12.25, 0.25), color="#8172B2", edgecolor="white")
ax[1, 1].set(title="Gap: bedtime - last eating (h)", xlabel="Hours")
for a in ax.flat:
    a.set_ylabel("Participants")
fig.tight_layout(); fig.savefig(FIG / "01_distributions.png", dpi=130); plt.close(fig)

# heaping check: how often times fall on the hour / half hour
log("\nHeaping: % last-eating times on :00 = "
    f"{100*np.isclose(df.last_50 % 1, 0).mean():.0f}%, on :00/:30 = "
    f"{100*np.isclose(df.last_50 % 0.5, 0).mean():.0f}%;  bedtimes on :00/:30 = "
    f"{100*np.isclose(df.bed_c % 0.5, 0).mean():.0f}%")

# --------------------------------------------------------------------------
# 6. Unadjusted: gap bins vs HbA1c
# --------------------------------------------------------------------------
bins = [-0.01, 1, 2, 3, 4, 5, 6, 12]
labels = ["0-1h", "1-2h", "2-3h", "3-4h", "4-5h", "5-6h", ">6h"]
df["gap_bin"] = pd.cut(df.gap, bins, labels=labels)
g = df.groupby("gap_bin", observed=True).apply(lambda x: pd.Series({
    "n": len(x),
    "w_pct_healthy": 100 * wmean(x.healthy, x.w),
    "w_mean_hba1c": wmean(x.LBXGH, x.w),
    "w_mean_age": wmean(x.RIDAGEYR, x.w),
    "w_mean_bmi": wmean(x.BMXBMI, x.w),
    "w_mean_bedtime": wmean(x.bed_c, x.w),
    "w_mean_kcal": wmean(x.DR1TKCAL, x.w),
}))
log("\n=== 4. GAP BINS (unadjusted, weighted) ===")
log(g.round(2).to_string())
g.to_csv(TAB / "gap_bins.csv")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].bar(g.index.astype(str), g.w_pct_healthy, color="#4C72B0")
ax[0].set(title="% with HbA1c < 5.7% by gap", ylabel="% (weighted)", ylim=(0, 100))
for i, (p, n) in enumerate(zip(g.w_pct_healthy, g.n)):
    ax[0].text(i, p + 1.5, f"n={int(n)}", ha="center", fontsize=8)
ax[1].bar(g.index.astype(str), g.w_mean_age, color="#DD8452")
ax[1].set(title="Mean age by gap (confounding)", ylabel="Years")
ax[2].bar(g.index.astype(str), g.w_mean_bedtime - 24, color="#55A868")
ax[2].set(title="Mean bedtime by gap", ylabel="Hours rel. to midnight")
fig.tight_layout(); fig.savefig(FIG / "02_gap_bins.png", dpi=130); plt.close(fig)

# --------------------------------------------------------------------------
# 7. Correlates of gap and of HbA1c
# --------------------------------------------------------------------------
cvars = ["gap", "LBXGH", "RIDAGEYR", "BMXBMI", "BMXWAIST", "INDFMPIR", "bed_c",
         "last_50", "sleep_h", "DR1TKCAL", "kcal_after20", "DR1TALCO"]
corr = df[cvars].corr(method="spearman")
corr.to_csv(TAB / "spearman_corr.csv")
log("\n=== 5. SPEARMAN CORRELATIONS with gap / HbA1c ===")
log(corr[["gap", "LBXGH"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cvars))); ax.set_xticklabels(cvars, rotation=60, ha="right")
ax.set_yticks(range(len(cvars))); ax.set_yticklabels(cvars)
for i in range(len(cvars)):
    for j in range(len(cvars)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im); ax.set_title("Spearman correlations")
fig.tight_layout(); fig.savefig(FIG / "03_correlations.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for lab, sub in df.groupby("race"):
    ax[0].plot(*sub.groupby(pd.cut(sub.RIDAGEYR, range(20, 85, 5)), observed=True)
               .LBXGH.mean().pipe(lambda s: (s.index.map(lambda iv: iv.mid), s.values)),
               marker="o", label=lab)
ax[0].set(title="HbA1c by age and race/ethnicity", xlabel="Age", ylabel="Mean HbA1c")
ax[0].legend(fontsize=8)
lo = sm.nonparametric.lowess(df.LBXGH, df.gap, frac=0.3)
ax[1].scatter(df.gap + np.random.uniform(-.1, .1, len(df)), df.LBXGH, s=3, alpha=.15)
ax[1].plot(lo[:, 0], lo[:, 1], color="crimson", lw=2, label="LOWESS")
ax[1].axhline(5.7, color="grey", ls="--")
ax[1].set(title="HbA1c vs gap (raw)", xlabel="Gap (h)", ylabel="HbA1c"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "04_age_race_and_raw_scatter.png", dpi=130); plt.close(fig)

# --------------------------------------------------------------------------
# 8. Adjusted models (weighted GLM, cluster-robust SE by strata x PSU)
#    Approximation of a full design-based analysis (use R `survey` for final).
# --------------------------------------------------------------------------
log("\n=== 6. ADJUSTED MODELS ===")
m = df.dropna(subset=["BMXBMI", "INDFMPIR", "educ", "sleep_h", "w"]).copy()
m = m[m.w > 0]
m["wn"] = m.w / m.w.mean()
log(f"Complete-case N = {len(m)} (dropped {len(df)-len(m)} for missing covariates)")
cov = ("RIDAGEYR + I(RIDAGEYR**2) + female + C(race) + BMXBMI + INDFMPIR + C(educ)"
       " + sleep_h + DR1TKCAL + bed_c")


def fit(formula, family, data=m):
    return smf.glm(formula, data, family=family, var_weights=data.wn).fit(
        cov_type="cluster", cov_kwds={"groups": data.cluster})


results = []
for name, fam, y in [("logistic: HbA1c<5.7", sm.families.Binomial(), "healthy"),
                     ("linear: HbA1c %", sm.families.Gaussian(), "LBXGH")]:
    base = fit(f"{y} ~ {cov}", fam)
    lin = fit(f"{y} ~ gap + {cov}", fam)
    spl = fit(f"{y} ~ bs(gap, df=4) + {cov}", fam)
    unadj = fit(f"{y} ~ gap", fam)
    b, se, p = lin.params["gap"], lin.bse["gap"], lin.pvalues["gap"]
    bu, pu = unadj.params["gap"], unadj.pvalues["gap"]
    wt = spl.wald_test(" , ".join(f"{c} = 0" for c in spl.params.index if "bs(gap" in c),
                       scalar=True)
    eff = f"OR/h={np.exp(b):.3f} [{np.exp(b-1.96*se):.3f}, {np.exp(b+1.96*se):.3f}]" \
        if "logistic" in name else f"beta/h={b:.4f} [{b-1.96*se:.4f}, {b+1.96*se:.4f}]"
    log(f"\n{name}")
    log(f"  unadjusted gap coef={bu:.4f} p={pu:.3g}")
    log(f"  adjusted   {eff}  p={p:.3g}")
    log(f"  spline (4 df) joint Wald p={float(wt.pvalue):.3g}")
    results.append((name, bu, pu, b, se, p, float(wt.pvalue)))
    if "logistic" in name:
        # predicted probability curve at covariate means / reference levels
        grid = pd.DataFrame({"gap": np.linspace(0, 8, 81)})
        ref = m.mode().iloc[0]
        for c in ["RIDAGEYR", "BMXBMI", "INDFMPIR", "sleep_h", "DR1TKCAL", "bed_c"]:
            grid[c] = m[c].mean()
        for c in ["female", "race", "educ"]:
            grid[c] = ref[c]
        sf = spl.get_prediction(grid).summary_frame()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(grid.gap, 100 * sf["mean"], color="#4C72B0", lw=2)
        ax.fill_between(grid.gap, 100 * sf.mean_ci_lower, 100 * sf.mean_ci_upper,
                        alpha=.25, color="#4C72B0")
        ax.set(title="Adjusted P(HbA1c < 5.7%) vs gap (spline, 95% CI)",
               xlabel="Bedtime - last eating occasion (h)", ylabel="Predicted %")
        ax2 = ax.twinx(); ax2.hist(m.gap, bins=32, range=(0, 8), alpha=.15, color="grey")
        ax2.set_yticks([])
        fig.tight_layout(); fig.savefig(FIG / "05_adjusted_spline.png", dpi=130); plt.close(fig)
        def auc(y, p):  # unweighted rank-based AUC
            r = pd.Series(p).rank().values
            n1 = y.sum(); n0 = len(y) - n1
            return (r[y.values == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        log(f"  AUC gap alone = {auc(m.healthy, unadj.predict()):.3f}; "
            f"covariates only = {auc(m.healthy, base.predict()):.3f}; "
            f"covariates + gap spline = {auc(m.healthy, spl.predict()):.3f}")

pd.DataFrame(results, columns=["model", "unadj_coef", "unadj_p", "adj_coef", "adj_se",
                               "adj_p", "spline_wald_p"]).to_csv(TAB / "models.csv", index=False)

# --------------------------------------------------------------------------
# 9. Sensitivity: alternative exposure definitions
# --------------------------------------------------------------------------
log("\n=== 7. SENSITIVITY (adjusted logistic, OR per hour) ===")
for var, lbl in [("gap", "last >=50 kcal occasion"),
                 ("gap_meal", "last main meal (excl. snacks/drinks)"),
                 ("gap_any", "last anything with kcal > 0")]:
    s = m[m[var].between(-1, 12)].copy()
    s[var] = s[var].clip(lower=0)
    r = fit(f"healthy ~ {var} + {cov}", sm.families.Binomial(), s)
    b, se = r.params[var], r.bse[var]
    log(f"  {lbl:40s} n={len(s):5d} OR={np.exp(b):.3f} "
        f"[{np.exp(b-1.96*se):.3f}, {np.exp(b+1.96*se):.3f}] p={r.pvalues[var]:.3g}")
r = fit(f"healthy ~ gap + {cov}", sm.families.Binomial(), m[m.DR1DAY.between(1, 5)])
log(f"  {'weekday recalls only':40s} n={int(m.DR1DAY.between(1,5).sum()):5d} "
    f"OR={np.exp(r.params['gap']):.3f} p={r.pvalues['gap']:.3g}")
r = fit(f"healthy ~ kcal_after20 + {cov}", sm.families.Binomial())
log(f"  {'kcal eaten after 20:00 (per 100 kcal)':40s} "
    f"OR={np.exp(100*r.params['kcal_after20']):.3f} p={r.pvalues['kcal_after20']:.3g}")

LOG.close()
print("\nDone. See outputs/")
