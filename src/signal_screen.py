"""
Signal screen: which plausible (non-diabetes) questions have strong, robust signal
in the files we have? Each question is adjusted for age, sex, race, education, income
(and smoking where relevant), design-based 95% CIs.

This is an exploratory screen to CHOOSE a project. The chosen project must then be
analysed with a pre-specified plan (otherwise it is p-hacking).

Run from project root:  python src/signal_screen.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from svy import Design  # noqa: E402

warnings.filterwarnings("ignore")
DATA = Path("data")
OUT = Path("outputs/signal_screen")
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "log.txt", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def read(name):
    df = pd.read_sas(DATA / f"{name}.xpt", encoding="infer")
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-70, 0.0)
    return df


def hhmm(s):
    try:
        h, m = str(s).split(":")
        return int(h) + int(m) / 60
    except Exception:
        return np.nan


demo = read("P_DEMO")
des = Design(demo)
df = demo
for f in ["P_SLQ", "P_BMX", "P_SMQ", "P_ALQ", "P_PAQ", "P_GHB", "P_DIQ"]:
    df = df.merge(read(f), on="SEQN", how="left")
tot = read("P_DR1TOT")[["SEQN", "DR1DRSTZ", "DR1TKCAL", "DR1TSUGR", "DR1TFIBE", "WTDRD1PP"]]
iff = read("P_DR1IFF")
iff = iff[(iff.DR1DRSTZ == 1) & (iff.DR1_020 >= 4 * 3600)]
late = iff[iff.DR1_020 >= 20 * 3600].groupby("SEQN").DR1IKCAL.sum()
df = df.merge(tot, on="SEQN", how="left")
df["kcal_after20"] = df.SEQN.map(late).fillna(0).where(df.DR1DRSTZ == 1)
df["pct_late"] = 100 * df.kcal_after20 / df.DR1TKCAL.where(df.DR1TKCAL > 0)

# --- cohort: adults, not pregnant, examined ---------------------------------
df = df[(df.RIDAGEYR >= 20) & (df.RIDEXPRG != 1) & (df.WTMECPRP > 0)].copy()
df["w"] = df.WTMECPRP
log(f"Adults (20+), not pregnant, examined: n = {len(df)}")

# --- derived variables ---------------------------------------------------------
df["female"] = (df.RIAGENDR == 2).astype(int)
df["race"] = df.RIDRETH3.map({1: "MexAm", 2: "OthHisp", 3: "NHWhite", 4: "NHBlack",
                              6: "NHAsian", 7: "Other"})
df["educ"] = df.DMDEDUC2.map({1: "<=HS", 2: "<=HS", 3: "<=HS", 4: "SomeColl", 5: "CollGrad"})
df["cur_smoke"] = np.select([df.SMQ020 == 2, df.SMQ040 == 3, df.SMQ040.isin([1, 2])],
                            [0, 0, 1], default=np.nan)
df["obese"] = (df.BMXBMI >= 30).astype(float).where(df.BMXBMI.notna())
df["sev_obese"] = (df.BMXBMI >= 35).astype(float).where(df.BMXBMI.notna())
df["sl_wd"], df["sl_we"] = df.SLD012, df.SLD013
df["sleep_avg"] = (5 * df.sl_wd + 2 * df.sl_we) / 7
df["sleep_cat"] = pd.cut(df.sleep_avg, [0, 5.99, 6.99, 9, 24],
                         labels=["<6h", "6-7h", "7-9h", ">9h"]).astype(str).replace("nan", np.nan)
bw, be = df.SLQ300.map(hhmm), df.SLQ320.map(hhmm)
bw, be = np.where(bw < 12, bw + 24, bw), np.where(be < 12, be + 24, be)
df["mid_wd"] = bw + df.sl_wd / 2
df["mid_we"] = be + df.sl_we / 2
df["sjl"] = (df.mid_we - df.mid_wd).abs()
df["sjl2"] = (df.sjl >= 2).astype(float).where(df.sjl.notna())
df["catchup"] = df.sl_we - df.sl_wd
df["catchup2"] = (df.catchup >= 2).astype(float).where(df.catchup.notna())
df["snore_freq"] = (df.SLQ030 == 3).astype(float).where(df.SLQ030.between(0, 3))
df["apnea_obs"] = (df.SLQ040 >= 2).astype(float).where(df.SLQ040.between(0, 3))
df["sleepy"] = (df.SLQ120 >= 3).astype(float).where(df.SLQ120.between(0, 4))
df["told_doc"] = (df.SLQ050 == 1).astype(float).where(df.SLQ050.isin([1, 2]))
FREQ_WK = {0: 0, 1: 7, 2: 5.5, 3: 3.5, 4: 2, 5: 1, 6: 2.5 / 4.33, 7: 1 / 4.33,
           8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52}
fw = np.where(df.ALQ111 == 2, 0, df.ALQ121.map(FREQ_WK))
dpw = np.where(fw == 0, 0, fw * df.ALQ130.where(df.ALQ130 < 777))
df["heavy_alc"] = pd.Series(np.where(df.female == 1, dpw > 7, dpw > 14), index=df.index
                            ).astype(float).where(~pd.isna(dpw))
# modified STOP-BANG (no blood pressure, no neck circumference): 6 items
items = pd.DataFrame({"S": df.snore_freq, "T": df.sleepy, "O": df.apnea_obs,
                      "B": df.sev_obese, "A": (df.RIDAGEYR > 50).astype(float),
                      "G": (df.female == 0).astype(float)})
df["sb6"] = items.sum(axis=1, min_count=6)
df["osa_hi"] = (df.sb6 >= 4).astype(float).where(df.sb6.notna())
df["hba1c_pre"] = (df.LBXGH >= 5.7).astype(float).where(df.LBXGH.notna())

BASE = " + RIDAGEYR + I(RIDAGEYR**2) + female + C(race) + C(educ) + INDFMPIR"


# --- screen -------------------------------------------------------------------
def fit_or(name, y, x_term, key, data, adj=BASE, weight=None):
    d = data.copy()
    if weight is not None:
        d["w"] = d[weight]
        d = d[d.w > 0]
    m = des.glm(f"{y} ~ {x_term}{adj}", d, "logit")
    m0 = des.glm(f"{y} ~ {x_term}", d, "logit")
    o, lo, hi = m.ci(key, exp=True)
    o0 = m0.ci(key, exp=True)[0]
    return dict(question=name, n=m.n, OR_unadj=o0, OR_adj=o, lo=lo, hi=hi, p=m.p[key])


rows = [
    fit_or("Frequent snoring -> obesity (BMI>=30)", "obese", "snore_freq", "snore_freq", df),
    fit_or("Observed apnea -> daytime sleepiness", "sleepy", "apnea_obs", "apnea_obs", df,
           BASE + " + BMXBMI"),
    fit_or("Short sleep <6h (vs 7-9h) -> obesity", "obese",
           "C(sleep_cat, Treatment('7-9h'))", "C(sleep_cat, Treatment('7-9h'))[T.<6h]", df),
    fit_or("Social jetlag >=2h -> obesity", "obese", "sjl2", "sjl2", df),
    fit_or("Weekend catch-up >=2h -> daytime sleepiness", "sleepy", "catchup2", "catchup2", df),
    fit_or("Current smoking -> frequent snoring", "snore_freq", "cur_smoke", "cur_smoke", df,
           BASE + " + BMXBMI"),
    fit_or("Heavy alcohol -> frequent snoring", "snore_freq", "heavy_alc", "heavy_alc", df,
           BASE + " + BMXBMI + cur_smoke"),
    fit_or("Short sleep <6h -> daytime sleepiness", "sleepy",
           "C(sleep_cat, Treatment('7-9h'))", "C(sleep_cat, Treatment('7-9h'))[T.<6h]", df),
    fit_or("High apnea risk (mod. STOP-BANG>=4) -> HbA1c>=5.7", "hba1c_pre", "osa_hi", "osa_hi",
           df[df.DIQ010 != 1]),
    fit_or("Observed apnea -> ever told doctor trouble sleeping", "told_doc", "apnea_obs",
           "apnea_obs", df),
]
scr = pd.DataFrame(rows)
log("\n=== SIGNAL SCREEN (adjusted OR, design-based 95% CI) ===")
log(scr.round(3).to_string(index=False))
scr.to_csv(OUT / "screen.csv", index=False)

# --- linear: late eating -> total calories (diet weights) -----------------------
d = df[(df.DR1DRSTZ == 1) & (df.WTDRD1PP > 0)].assign(w=lambda x: x.WTDRD1PP)
d["late_cat"] = pd.cut(d.pct_late, [-1, 0, 25, 100], labels=["0%", "1-25%", ">25%"]).astype(str)
m = des.glm(f"DR1TKCAL ~ C(late_cat, Treatment('0%')){BASE}", d)
for lv in ["1-25%", ">25%"]:
    k = f"C(late_cat, Treatment('0%'))[T.{lv}]"
    e = m.ci(k)
    log(f"Late eating {lv} of kcal after 20:00 vs none -> +{e[0]:.0f} kcal/day "
        f"({e[1]:.0f}, {e[2]:.0f}) p={m.p[k]:.2g}")

# --- the care-gap descriptive -------------------------------------------------
log("\n=== APNEA CARE GAP (descriptive, weighted) ===")
hi = df[df.osa_hi == 1]
log(f"Modified STOP-BANG computable: n={df.sb6.notna().sum()}; "
    f"high-risk (>=4 of 6): n={len(hi)}, weighted {100*des.mean(df.osa_hi, df)[0]:.1f}% "
    f"(~{df.loc[df.osa_hi == 1, 'w'].sum()/1e6:.1f} M US adults)")
sub = df.assign(w=df.w.where(df.osa_hi == 1, 0))
e = des.mean(df.told_doc.where(df.osa_hi == 1), sub)
log(f"Of high-risk: % EVER told a doctor about trouble sleeping: {100*e[0]:.1f}% "
    f"({100*e[1]:.1f}-{100*e[2]:.1f})  -> never told: {100-100*e[0]:.1f}%")
gap = []
for g, lab in [("female", "sex"), ("race", "race"), ("educ", "education")]:
    for lv in sorted(df[g].dropna().unique()):
        msk = (df.osa_hi == 1) & (df[g] == lv)
        if msk.sum() < 30:
            continue
        s = df.assign(w=df.w.where(msk, 0))
        e = des.mean(df.told_doc.where(msk), s)
        gap.append((lab, lv, int(msk.sum()), 100 * e[0], 100 * e[1], 100 * e[2]))
gap = pd.DataFrame(gap, columns=["by", "group", "n_highrisk", "pct_told_doc", "lo", "hi"])
log(gap.round(1).to_string(index=False))
gap.to_csv(OUT / "care_gap.csv", index=False)
m = des.glm(f"told_doc ~ female + C(race, Treatment('NHWhite')) + C(educ) + INDFMPIR"
            f" + RIDAGEYR", df[df.osa_hi == 1], "logit")
log("Adjusted OR of having told a doctor, among high-risk:")
for k in m.b.index[1:]:
    o, lo, hi_ = m.ci(k, exp=True)
    log(f"   {k:40s} {o:.2f} ({lo:.2f}-{hi_:.2f}) p={m.p[k]:.3g}")
LOG.close()
