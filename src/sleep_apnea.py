"""
Are snoring, snorting/breathing pauses and daytime sleepiness associated with HbA1c
in non-diabetic US adults?  NHANES 2017-March 2020 pre-pandemic.

Run from project root:  python src/sleep_apnea.py
Outputs: outputs/sleep_apnea/{figures,tables}/, outputs/sleep_apnea/log.txt

Variance: design-based Taylor linearization (strata x PSU, with-replacement,
same as R survey::svyglm default), subpopulation handled by keeping all PSUs.
Weights: MEC exam weights WTMECPRP.
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
from scipy import stats

warnings.filterwarnings("ignore")
DATA = Path("data")
OUT = Path("outputs/sleep_apnea")
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "log.txt", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def read(name):
    df = pd.read_sas(DATA / f"{name}.xpt", encoding="infer")
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-70, 0.0)  # SAS tiny-float zeros
    return df


# --------------------------------------------------------------------------
# Survey design machinery
# --------------------------------------------------------------------------
demo = read("P_DEMO")
DESIGN = demo.loc[demo.WTMECPRP > 0, ["SDMVSTRA", "SDMVPSU"]].drop_duplicates()
N_PSU, N_STRATA = len(DESIGN), DESIGN.SDMVSTRA.nunique()
DDF = N_PSU - N_STRATA  # design degrees of freedom


def svy_vcov(res, data):
    """Taylor-linearized vcov for a GLM with canonical link fit with var_weights."""
    X = res.model.exog
    w = data["w"].values
    mu = res.fittedvalues.values if hasattr(res.fittedvalues, "values") else res.fittedvalues
    y = res.model.endog
    fam = res.model.family
    v = mu * (1 - mu) if isinstance(fam, sm.families.Binomial) else np.ones_like(mu)
    H = (X * (w * v)[:, None]).T @ X
    U = X * (w * (y - mu))[:, None]
    Udf = pd.DataFrame(U)
    Udf["s"] = data.SDMVSTRA.values
    Udf["p"] = data.SDMVPSU.values
    tot = Udf.groupby(["s", "p"]).sum()
    tot = tot.reindex(pd.MultiIndex.from_frame(DESIGN.rename(columns={"SDMVSTRA": "s",
                                                                      "SDMVPSU": "p"})),
                      fill_value=0.0)  # subpopulation: absent PSUs contribute zero
    M = np.zeros((X.shape[1], X.shape[1]))
    for _, g in tot.groupby(level="s"):
        n_h = len(g)
        d = g.values - g.values.mean(axis=0)
        M += n_h / (n_h - 1) * d.T @ d
    Hi = np.linalg.inv(H)
    return pd.DataFrame(Hi @ M @ Hi, index=res.params.index, columns=res.params.index)


class Svy:
    def __init__(self, formula, data, family):
        self.data = data
        self.res = smf.glm(formula, data, family=family, var_weights=data.w).fit()
        used = data.loc[self.res.model.data.row_labels]  # rows patsy kept (NaN dropped)
        self.n = len(used)
        self.V = svy_vcov(self.res, used)
        self.b = self.res.params
        self.se = np.sqrt(np.diag(self.V))
        self.se = pd.Series(self.se, index=self.b.index)
        self.p = pd.Series(2 * stats.t.sf(np.abs(self.b / self.se), DDF), index=self.b.index)

    def ci(self, term, exp=False):
        q = stats.t.ppf(0.975, DDF)
        b, se = self.b[term], self.se[term]
        out = (b, b - q * se, b + q * se)
        return tuple(np.exp(out)) if exp else out

    def wald(self, terms):
        """Joint test; F with design df (like survey::regTermTest)."""
        idx = [t for t in self.b.index if any(t.startswith(x) for x in terms)]
        b = self.b[idx].values
        V = self.V.loc[idx, idx].values
        q = len(idx)
        F = float(b @ np.linalg.solve(V, b)) / q
        return stats.f.sf(F, q, DDF - q + 1)


def svy_mean(x, data):
    """Weighted mean/proportion with design-based 95% CI."""
    d = data.assign(_y=x).dropna(subset=["_y"])
    m = Svy("_y ~ 1", d, sm.families.Gaussian())
    est, lo, hi = m.ci("Intercept")
    return est, lo, hi


# --------------------------------------------------------------------------
# 1. Load + derive variables
# --------------------------------------------------------------------------
files = ["P_DEMO", "P_GHB", "P_DIQ", "P_SLQ", "P_BMX", "P_SMQ", "P_ALQ", "P_PAQ"]
raw = {f: read(f) for f in files}
log("=== 1. FILES ===")
for f, d in raw.items():
    log(f"{f:7s} rows={len(d):6d} cols={d.shape[1]:3d}")
log(f"Design: {N_STRATA} strata, {N_PSU} PSUs, design df = {DDF}")

df = raw["P_DEMO"]
for f in files[1:]:
    df = df.merge(raw[f], on="SEQN", how="left")

# Exposures ---------------------------------------------------------------
SNORE = {0: "Never", 1: "Rarely", 2: "Occasionally", 3: "Frequently", 9: "Don't know"}
PAUSE = {0: "Never", 1: "Rarely", 2: "Occasionally", 3: "Frequently", 9: "Don't know"}
SLEEPY = {0: "Never", 1: "Rarely", 2: "Sometimes", 3: "Often", 4: "Almost always"}
df["snore"] = df.SLQ030.map(SNORE)
df["pause"] = df.SLQ040.map(PAUSE)
df["sleepy"] = df.SLQ120.map(SLEEPY)
# ordinal versions for trend tests (don't-know -> NaN)
df["snore_o"] = df.SLQ030.where(df.SLQ030.between(0, 3))
df["pause_o"] = df.SLQ040.where(df.SLQ040.between(0, 3))
df["sleepy_o"] = df.SLQ120.where(df.SLQ120.between(0, 4))
# binary "high" symptoms
df["snore_hi"] = (df.SLQ030 == 3).astype(float).where(df.SLQ030.between(0, 3))
df["pause_hi"] = (df.SLQ040 >= 2).astype(float).where(df.SLQ040.between(0, 3))
df["sleepy_hi"] = (df.SLQ120 >= 3).astype(float).where(df.SLQ120.between(0, 4))
df["score"] = df[["snore_hi", "pause_hi", "sleepy_hi"]].sum(axis=1, min_count=3)
df["score_c"] = df.score.clip(upper=2).map({0: "0", 1: "1", 2: "2-3"})

# Outcome -------------------------------------------------------------------
df["prediab"] = (df.LBXGH >= 5.7).astype(float)

# Covariates ----------------------------------------------------------------
df["female"] = (df.RIAGENDR == 2).astype(int)
df["race"] = df.RIDRETH3.map({1: "MexAm", 2: "OthHisp", 3: "NHWhite", 4: "NHBlack",
                              6: "NHAsian", 7: "Other"})
df["educ"] = df.DMDEDUC2.map({1: "<=HS", 2: "<=HS", 3: "<=HS", 4: "SomeColl", 5: "CollGrad"})
df["smoke"] = np.select([df.SMQ020 == 2, (df.SMQ020 == 1) & (df.SMQ040 == 3),
                         (df.SMQ020 == 1) & df.SMQ040.isin([1, 2])],
                        ["Never", "Former", "Current"], default=None)
# alcohol: drinks/week = frequency/week x drinks/day
FREQ_WK = {0: 0, 1: 7, 2: 5.5, 3: 3.5, 4: 2, 5: 1, 6: 2.5 / 4.33, 7: 1 / 4.33,
           8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52}
fw = df.ALQ121.map(FREQ_WK)
fw = np.where(df.ALQ111 == 2, 0, fw)
dpd = df.ALQ130.where(df.ALQ130 < 777)
dpw = np.where(fw == 0, 0, fw * dpd)
df["drinks_wk"] = dpw
heavy_cut = np.where(df.female == 1, 7, 14)
df["alcohol"] = np.select([df.drinks_wk == 0, df.drinks_wk <= heavy_cut,
                           df.drinks_wk > heavy_cut], ["None", "Moderate", "Heavy"],
                          default=None)
df.loc[pd.isna(df.drinks_wk), "alcohol"] = None


def mins(days, m):
    d = df[days].where(df[days] <= 7)
    mm = df[m].where(df[m] < 7777)
    return (d * mm).fillna(0).where(~(d.isna() & df[days].isin([77, 99])))


# MVPA-equivalent minutes/week (vigorous counts double); "no" answers -> 0
df["mvpa"] = (2 * mins("PAQ610", "PAD615") + mins("PAQ625", "PAD630")
              + mins("PAQ640", "PAD645") + 2 * mins("PAQ655", "PAD660")
              + mins("PAQ670", "PAD675"))
df.loc[df.PAQ605.isna() & df.PAQ650.isna(), "mvpa"] = np.nan
df["pa"] = pd.cut(df.mvpa, [-1, 0, 149.99, 1e9], labels=["Inactive", "Insufficient",
                                                          "Active"]).astype(object)
df["sleep_h"] = (5 * df.SLD012 + 2 * df.SLD013) / 7
df["obese"] = (df.BMXBMI >= 30).astype(float).where(df.BMXBMI.notna())
df["age50"] = (df.RIDAGEYR >= 50).astype(int)
df["w"] = df.WTMECPRP

# --------------------------------------------------------------------------
# 2. Cohort
# --------------------------------------------------------------------------
funnel = [("All P_DEMO", len(df))]


def step(label, mask):
    global df
    df = df[mask]
    funnel.append((label, len(df)))


step("Examined (MEC weight > 0)", df.w > 0)
step("Age >= 20", df.RIDAGEYR >= 20)
step("Not pregnant", df.RIDEXPRG != 1)
step("HbA1c measured", df.LBXGH.notna())
step("No diagnosed diabetes", df.DIQ010 != 1)
step("No insulin / diabetes pills", (df.DIQ050 != 1) & (df.DIQ070 != 1))
step("HbA1c < 6.5% (excl. undiagnosed diabetes)", df.LBXGH < 6.5)
ELIG = df.copy()
step("Valid answer on all 3 symptom questions (DK kept as category)",
     df.snore.notna() & df.pause.notna() & df.sleepy.notna())

COVS = ["RIDAGEYR", "female", "race", "educ", "INDFMPIR", "BMXBMI", "smoke",
        "alcohol", "pa"]
miss = df[COVS + ["BMXWAIST", "sleep_h"]].isna().mean().mul(100).round(1)
step("Complete covariates (common sample for M0-M3)", df[COVS].notna().all(axis=1))
log("\n=== 2. COHORT FUNNEL ===")
for lbl, n in funnel:
    log(f"{n:7d}  {lbl}")
log("\n% missing among symptom responders:\n" + miss.to_string())
pd.DataFrame(funnel, columns=["step", "n"]).to_csv(TAB / "cohort_funnel.csv", index=False)
df.to_csv(OUT / "analytic_cohort.csv", index=False)

# --------------------------------------------------------------------------
# 3. Symptom prevalence + Table 1
# --------------------------------------------------------------------------
log("\n=== 3. WEIGHTED SYMPTOM PREVALENCE (complete-case cohort) ===")
prev = []
for var, levels in [("snore", SNORE), ("pause", PAUSE), ("sleepy", SLEEPY)]:
    for lvl in levels.values():
        n = int((df[var] == lvl).sum())
        if n == 0:
            continue
        est, lo, hi = svy_mean((df[var] == lvl).astype(float), df)
        pe, plo, phi = svy_mean(df.prediab.where(df[var] == lvl), df.assign(
            w=df.w.where(df[var] == lvl, 0)))
        prev.append((var, lvl, n, 100 * est, 100 * pe, 100 * plo, 100 * phi))
prev = pd.DataFrame(prev, columns=["symptom", "level", "n", "w_pct_of_pop",
                                   "w_pct_prediab", "lo", "hi"])
log(prev.round(1).to_string(index=False))
prev.to_csv(TAB / "symptom_prevalence.csv", index=False)


def t1(group):
    rows = {}
    for lvl, g in df.groupby(group):
        g2 = df.assign(w=df.w.where(df[group] == lvl, 0))
        col = {"n": len(g)}
        for v, lab in [("RIDAGEYR", "Age, y"), ("BMXBMI", "BMI"), ("BMXWAIST", "Waist, cm"),
                       ("INDFMPIR", "Income/poverty"), ("sleep_h", "Sleep h"),
                       ("LBXGH", "HbA1c %"), ("mvpa", "MVPA min/wk")]:
            col[lab] = svy_mean(g2[v].where(df[group] == lvl), g2)[0]
        for v, lab in [("female", "% female"), ("obese", "% obese"),
                       ("prediab", "% HbA1c>=5.7")]:
            col[lab] = 100 * svy_mean(g2[v].where(df[group] == lvl), g2)[0]
        for v, cat, lab in [("smoke", "Current", "% current smoker"),
                            ("alcohol", "Heavy", "% heavy drinker"),
                            ("pa", "Active", "% meets PA guideline")]:
            col[lab] = 100 * svy_mean((g2[v] == cat).astype(float).where(df[group] == lvl),
                                      g2)[0]
        rows[lvl] = col
    return pd.DataFrame(rows)


table1 = t1("score_c")
log("\n=== 4. TABLE 1 by symptom score (weighted) ===")
log(table1.round(1).to_string())
table1.to_csv(TAB / "table1_by_score.csv")

# --------------------------------------------------------------------------
# 4. Main models
# --------------------------------------------------------------------------
M = {
    "M0 unadjusted": "",
    "M1 +demographics": " + RIDAGEYR + I(RIDAGEYR**2) + female + C(race) + C(educ) + INDFMPIR",
}
M["M2 +BMI"] = M["M1 +demographics"] + " + BMXBMI"
M["M3 +smoking/alcohol/PA"] = M["M2 +BMI"] + " + C(smoke) + C(alcohol) + C(pa)"

EXPO = {
    "snore": ("C(snore, Treatment('Never'))", "snore_o",
              ["Rarely", "Occasionally", "Frequently", "Don't know"]),
    "pause": ("C(pause, Treatment('Never'))", "pause_o",
              ["Rarely", "Occasionally", "Frequently", "Don't know"]),
    "sleepy": ("C(sleepy, Treatment('Never'))", "sleepy_o",
               ["Rarely", "Sometimes", "Often", "Almost always"]),
    "score": ("C(score_c, Treatment('0'))", "score", ["1", "2-3"]),
}

rows = []
log("\n=== 5. MODELS: OR for HbA1c >= 5.7% and beta for HbA1c (%) ===")
for ename, (term, ordv, lvls) in EXPO.items():
    log(f"\n--- {ename} ---")
    for mname, adj in M.items():
        lg = Svy(f"prediab ~ {term}{adj}", df, sm.families.Binomial())
        ln = Svy(f"LBXGH ~ {term}{adj}", df, sm.families.Gaussian())
        dd = df.dropna(subset=[ordv])
        tr = Svy(f"prediab ~ {ordv}{adj}", dd, sm.families.Binomial())
        trl = Svy(f"LBXGH ~ {ordv}{adj}", dd, sm.families.Gaussian())
        p_all = lg.wald([term])
        parts = []
        for lv in lvls:
            k = f"{term}[T.{lv}]"
            orr, lo, hi = lg.ci(k, exp=True)
            b, blo, bhi = ln.ci(k)
            rows.append((ename, mname, lv, int((df[ename if ename != 'score' else 'score_c']
                                                == lv).sum()),
                         orr, lo, hi, lg.p[k], b, blo, bhi, ln.p[k]))
            parts.append(f"{lv}: OR {orr:.2f} ({lo:.2f}-{hi:.2f})")
        log(f"{mname:24s} " + "; ".join(parts))
        log(f"{'':24s} p(overall)={p_all:.3g}  p-trend OR={tr.p[ordv]:.3g} "
            f"(OR/step {np.exp(tr.b[ordv]):.3f});  HbA1c beta/step={trl.b[ordv]:.4f} "
            f"p={trl.p[ordv]:.3g}")
res = pd.DataFrame(rows, columns=["exposure", "model", "level", "n", "OR", "OR_lo", "OR_hi",
                                  "OR_p", "beta_hba1c", "beta_lo", "beta_hi", "beta_p"])
res.to_csv(TAB / "models_all.csv", index=False)

# --------------------------------------------------------------------------
# 5. Effect modification (score, fully adjusted)
# --------------------------------------------------------------------------
log("\n=== 6. EFFECT MODIFICATION (score continuous 0-3, M3) ===")
adj3 = M["M3 +smoking/alcohol/PA"]
em = []
for mod, lab in [("female", "sex (female vs male)"), ("obese", "obesity (BMI>=30)"),
                 ("age50", "age >= 50")]:
    d = df.dropna(subset=[mod])
    r = Svy(f"prediab ~ score * {mod}{adj3}", d, sm.families.Binomial())
    pint = r.p[f"score:{mod}"]
    out = []
    for val in [0, 1]:
        dv = d.assign(w=d.w.where(d[mod] == val, 0))
        adj_s = adj3.replace(" + female", "") if mod == "female" else adj3
        rv = Svy(f"prediab ~ score{adj_s}", dv[dv.w > 0], sm.families.Binomial())
        orr, lo, hi = rv.ci("score", exp=True)
        out.append(f"{mod}={val}: OR/symptom {orr:.2f} ({lo:.2f}-{hi:.2f})")
        em.append((lab, val, orr, lo, hi, pint))
    log(f"{lab:24s} " + "; ".join(out) + f";  p-interaction={pint:.3g}")
pd.DataFrame(em, columns=["modifier", "value", "OR", "lo", "hi", "p_int"]).to_csv(
    TAB / "effect_modification.csv", index=False)

# --------------------------------------------------------------------------
# 6. Sensitivity analyses (score continuous, M3 unless stated)
# --------------------------------------------------------------------------
log("\n=== 7. SENSITIVITY (OR per additional symptom, 0-3) ===")


def sens(label, data, formula_adj):
    r = Svy(f"prediab ~ score{formula_adj}", data, sm.families.Binomial())
    orr, lo, hi = r.ci("score", exp=True)
    log(f"  {label:48s} n={r.n:5d} OR={orr:.3f} ({lo:.3f}-{hi:.3f}) p={r.p['score']:.3g}")


sens("Main (M3)", df, adj3)
sens("M3 + waist instead of BMI", df.dropna(subset=["BMXWAIST"]),
     adj3.replace("BMXBMI", "BMXWAIST"))
sens("M3 + BMI + waist", df.dropna(subset=["BMXWAIST"]), adj3 + " + BMXWAIST")
sens("M3 + sleep duration (possible mediator)", df.dropna(subset=["sleep_h"]),
     adj3 + " + sleep_h + I(sleep_h**2)")
sens("M3 excluding borderline diabetes (DIQ010=3)", df[df.DIQ010 != 3], adj3)
sens("M3 excluding obese", df[df.obese == 0], adj3)
# income missing -> indicator instead of dropping
e = ELIG[ELIG.snore.notna() & ELIG.pause.notna() & ELIG.sleepy.notna()].copy()
e["pir_miss"] = e.INDFMPIR.isna().astype(int)
e["INDFMPIR"] = e.INDFMPIR.fillna(e.INDFMPIR.median())
e = e.dropna(subset=[c for c in COVS if c != "INDFMPIR"])
sens("M3 with income-missing indicator (larger n)", e, adj3 + " + pir_miss")

# --------------------------------------------------------------------------
# 7. Figures
# --------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
for a, var in zip(ax, ["snore", "pause", "sleepy"]):
    p = prev[prev.symptom == var]
    a.bar(p.level, p.w_pct_prediab, color="#4C72B0",
          yerr=[p.w_pct_prediab - p.lo, p.hi - p.w_pct_prediab], capsize=3)
    for i, (y, n) in enumerate(zip(p.w_pct_prediab, p.n)):
        a.text(i, 2, f"n={n}", ha="center", fontsize=8, color="white")
    a.set_title({"snore": "Snoring", "pause": "Snorting / breathing pauses",
                 "sleepy": "Daytime sleepiness"}[var])
    a.tick_params(axis="x", rotation=30)
ax[0].set_ylabel("% with HbA1c >= 5.7% (weighted, 95% CI)")
fig.suptitle("Unadjusted prevalence of prediabetes-range HbA1c by symptom")
fig.tight_layout(); fig.savefig(FIG / "01_prediab_by_symptom.png", dpi=130); plt.close(fig)

# forest plot: top category vs reference across models
top = {"snore": "Frequently", "pause": "Frequently", "sleepy": "Almost always", "score": "2-3"}
lab = {"snore": "Snoring: frequent vs never", "pause": "Pauses: frequent vs never",
       "sleepy": "Sleepy: almost always vs never", "score": "Symptom score: 2-3 vs 0"}
fig, ax = plt.subplots(figsize=(8, 5.5))
y, ticks, tl = 0, [], []
colors = ["#999999", "#DD8452", "#4C72B0", "#C44E52"]
for ename in ["snore", "pause", "sleepy", "score"]:
    sub = res[(res.exposure == ename) & (res.level == top[ename])]
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.errorbar(r.OR, y, xerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt="o",
                    color=colors[i], capsize=3, label=r.model if ename == "snore" else None)
        y -= 1
    ticks.append(y + 2.5); tl.append(lab[ename]); y -= 1
ax.axvline(1, color="k", lw=.8)
ax.set_xscale("log"); ax.set_yticks(ticks); ax.set_yticklabels(tl)
ax.set_xlabel("Odds ratio for HbA1c >= 5.7% (95% CI, design-based)")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("How adjustment changes the association")
fig.tight_layout(); fig.savefig(FIG / "02_forest_models.png", dpi=130); plt.close(fig)

# adjusted predicted HbA1c by score (M3), marginal standardization
r = Svy(f"LBXGH ~ C(score_c, Treatment('0')){adj3}", df, sm.families.Gaussian())
r2 = Svy(f"prediab ~ C(score_c, Treatment('0')){adj3}", df, sm.families.Binomial())
lv = ["0", "1", "2-3"]
pm = [float(np.average(r.res.predict(df.assign(score_c=l)), weights=df.w)) for l in lv]
pp = [float(100 * np.average(r2.res.predict(df.assign(score_c=l)), weights=df.w)) for l in lv]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(lv, pm, color="#4C72B0"); ax[0].set_ylim(min(pm) - .1, max(pm) + .05)
ax[0].set(title="Adjusted mean HbA1c (M3)", xlabel="Number of high symptoms", ylabel="HbA1c %")
ax[1].bar(lv, pp, color="#C44E52")
ax[1].set(title="Adjusted % HbA1c >= 5.7% (M3)", xlabel="Number of high symptoms", ylabel="%")
for a, vals, f in [(ax[0], pm, "{:.2f}"), (ax[1], pp, "{:.1f}")]:
    for i, v in enumerate(vals):
        a.text(i, v, f.format(v), ha="center", va="bottom")
fig.tight_layout(); fig.savefig(FIG / "03_adjusted_by_score.png", dpi=130); plt.close(fig)
log(f"\nAdjusted (M3) mean HbA1c by score 0/1/2-3: {[round(x, 3) for x in pm]}")
log(f"Adjusted (M3) % HbA1c>=5.7 by score 0/1/2-3: {[round(x, 1) for x in pp]}")

LOG.close()
print("\nDone. See outputs/sleep_apnea/")
