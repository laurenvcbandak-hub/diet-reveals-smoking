"""
Coffee signal screen: which coffee-related questions have strong, robust signal?
NHANES 2017-March 2020. Diet day-1 weights, design-based 95% CIs.
Exploratory: used to CHOOSE a project, not as the final analysis.

Food codes (FNDDS): 921 coffee (9211 decaf), 923 tea, 924 soft drinks, 953 energy drinks.
Run from project root:  python src/coffee_screen.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from svy import Design  # noqa: E402

warnings.filterwarnings("ignore")
DATA = Path("data")
OUT = Path("outputs/coffee_screen")
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


demo = read("P_DEMO")
des = Design(demo)

# ---------------- beverage exposures from food codes -------------------------
iff = read("P_DR1IFF")
iff = iff[iff.DR1DRSTZ == 1].copy()
code = iff.DR1IFDCD.astype(int)
p3, p4 = code // 100000, code // 10000
dens = 100 * iff.DR1IKCAL / iff.DR1IGRMS.where(iff.DR1IGRMS > 0)
iff["src"] = np.select([p4 == 9211, p3 == 921, p3 == 923, (p3 == 924) & (dens < 10),
                        p3 == 924, p3 == 953],
                       ["decaf", "coffee", "tea", "dietsoda", "regsoda", "energy"],
                       default="other")
iff["caf"] = iff.DR1ICAFF.fillna(0)
g = iff.groupby(["SEQN", "src"])
caf = g.caf.sum().unstack(fill_value=0)
grams = g.DR1IGRMS.sum().unstack(fill_value=0)
kcal = g.DR1IKCAL.sum().unstack(fill_value=0)
sugar = g.DR1ISUGR.sum().unstack(fill_value=0)
per = pd.DataFrame(index=iff.SEQN.unique())
for s in ["coffee", "tea", "regsoda", "dietsoda", "energy"]:
    per[f"caf_{s}"] = caf.get(s, 0)
per["caf_soda"] = per.caf_regsoda + per.caf_dietsoda
per["coffee_cups"] = (grams.get("coffee", 0) / 240)       # 1 cup = 240 g (8 oz)
per["decaf_cups"] = grams.get("decaf", 0) / 240
per["coffee_kcal"] = kcal.get("coffee", 0) + kcal.get("decaf", 0)
per["coffee_sugar"] = sugar.get("coffee", 0) + sugar.get("decaf", 0)
per = per.fillna(0)
per.index.name = "SEQN"

# ---------------- merge ---------------------------------------------------------
df = demo.merge(read("P_DR1TOT")[["SEQN", "DR1DRSTZ", "DR1TKCAL", "DR1TSUGR", "DR1TFIBE",
                                   "WTDRD1PP"]], on="SEQN")
for f in ["P_SLQ", "P_BMX", "P_SMQ", "P_ALQ", "P_PAQ", "P_GHB", "P_DIQ"]:
    df = df.merge(read(f), on="SEQN", how="left")
df = df.merge(per, left_on="SEQN", right_index=True, how="left")
df = df[(df.RIDAGEYR >= 20) & (df.RIDEXPRG != 1) & (df.DR1DRSTZ == 1) & (df.WTDRD1PP > 0)].copy()
df["w"] = df.WTDRD1PP
log(f"Adults with reliable day-1 recall: n = {len(df)}")

df["female"] = (df.RIAGENDR == 2).astype(int)
df["race"] = df.RIDRETH3.map({1: "MexAm", 2: "OthHisp", 3: "NHWhite", 4: "NHBlack",
                              6: "NHAsian", 7: "Other"})
df["educ"] = df.DMDEDUC2.map({1: "<=HS", 2: "<=HS", 3: "<=HS", 4: "SomeColl", 5: "CollGrad"})
df["cur_smoke"] = pd.Series(np.select([df.SMQ020 == 2, df.SMQ040 == 3, df.SMQ040.isin([1, 2])],
                                      [0, 0, 1], default=-1), index=df.index
                            ).replace(-1, np.nan).astype(float)
df["obese"] = (df.BMXBMI >= 30).astype(float).where(df.BMXBMI.notna())
df["prediab"] = (df.LBXGH >= 5.7).astype(float).where(df.LBXGH.notna() & (df.DIQ010 != 1)
                                                       & (df.DIQ050 != 1) & (df.DIQ070 != 1))
df["sleepy"] = (df.SLQ120 >= 3).astype(float).where(df.SLQ120.between(0, 4))
df["sleep_avg"] = (5 * df.SLD012 + 2 * df.SLD013) / 7
df["short_sleep"] = (df.sleep_avg < 7).astype(float).where(df.sleep_avg.notna())
FREQ_WK = {0: 0, 1: 7, 2: 5.5, 3: 3.5, 4: 2, 5: 1, 6: 2.5 / 4.33, 7: 1 / 4.33,
           8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52}
fw = np.where(df.ALQ111 == 2, 0, df.ALQ121.map(FREQ_WK))
dpw = np.where(fw == 0, 0, fw * df.ALQ130.where(df.ALQ130 < 777))
df["heavy_alc"] = pd.Series(np.where(df.female == 1, dpw > 7, dpw > 14), index=df.index
                            ).astype(float).where(~pd.isna(dpw))
df["noncoffee_sugar"] = df.DR1TSUGR - df.coffee_sugar
df["cups_cat"] = pd.cut(df.coffee_cups, [-1, 0.01, 1, 2, 4, 100],
                        labels=["0", "<1", "1-2", "2-4", ">4"]).astype(str)
df["sweet"] = np.select([df.coffee_cups < 0.01, df.coffee_sugar < 5, df.coffee_sugar >= 5],
                        ["No coffee", "Black/low-sugar", "Sweetened (>=5 g sugar)"], default="")
df["sweet"] = df.sweet.replace("", np.nan)

# ---------------- descriptives --------------------------------------------------
log("\n=== WHO DRINKS COFFEE (weighted) ===")
log(f"% drinking caffeinated coffee: {100*des.mean((df.coffee_cups > .01).astype(float), df)[0]:.1f}%")
for lv in ["0", "<1", "1-2", "2-4", ">4"]:
    m = df.cups_cat == lv
    s = df.assign(w=df.w.where(m, 0))
    log(f"  {lv:>4s} cups: n={m.sum():5d}  age {des.mean(df.RIDAGEYR.where(m), s)[0]:.0f}  "
        f"smoker {100*des.mean(df.cur_smoke.where(m), s)[0]:.0f}%  "
        f"obese {100*des.mean(df.obese.where(m), s)[0]:.0f}%  "
        f"prediab {100*des.mean(df.prediab.where(m), s)[0]:.0f}%")
d = df[df.coffee_cups > .01]
s_ = d.assign(w=d.w)
log(f"Among coffee drinkers: mean coffee kcal {des.mean(d.coffee_kcal, s_)[0]:.0f}, "
    f"mean sugar from coffee {des.mean(d.coffee_sugar, s_)[0]:.1f} g; "
    f"% sweetened (>=5 g): {100*des.mean((d.coffee_sugar >= 5).astype(float), s_)[0]:.0f}%")

# ---------------- screen ----------------------------------------------------------
BASE = " + RIDAGEYR + I(RIDAGEYR**2) + female + C(race) + C(educ) + INDFMPIR"
BASE_S = BASE + " + cur_smoke"
rows = []


def binscreen(name, y, x, key, adj, data=df):
    m = des.glm(f"{y} ~ {x}{adj}", data, "logit")
    m0 = des.glm(f"{y} ~ {x}", data, "logit")
    o, lo, hi = m.ci(key, exp=True)
    rows.append(dict(question=name, n=m.n, effect="OR", unadj=m0.ci(key, exp=True)[0],
                     adj=o, lo=lo, hi=hi, p=m.p[key]))


def linscreen(name, y, x, key, adj, scale=1, data=df):
    m = des.glm(f"{y} ~ {x}{adj}", data)
    m0 = des.glm(f"{y} ~ {x}", data)
    e = m.ci(key, scale=scale)
    rows.append(dict(question=name, n=m.n, effect="diff", unadj=m0.ci(key, scale=scale)[0],
                     adj=e[0], lo=e[1], hi=e[2], p=m.p[key]))


CUPS = "C(cups_cat, Treatment('0'))"
K = f"{CUPS}[T.2-4]"
binscreen("2-4 cups vs none -> current smoker", "cur_smoke", CUPS, K, BASE)
binscreen("2-4 cups vs none -> heavy alcohol", "heavy_alc", CUPS, K, BASE_S)
binscreen("2-4 cups vs none -> obesity", "obese", CUPS, K, BASE_S)
binscreen("2-4 cups vs none -> prediabetes HbA1c", "prediab", CUPS, K, BASE_S + " + BMXBMI")
binscreen("2-4 cups vs none -> often sleepy", "sleepy", CUPS, K, BASE_S)
binscreen("2-4 cups vs none -> short sleep <7h", "short_sleep", CUPS, K, BASE_S)
SRC = " + caf_coffee + caf_tea + caf_energy"
binscreen("Soda caffeine (per 100 mg, coffee/tea held) -> obesity", "obese",
          "caf_soda", "caf_soda", BASE_S + " + caf_coffee + caf_tea + caf_energy")
binscreen("Coffee caffeine (per 100 mg, soda/tea held) -> obesity", "obese",
          "caf_coffee", "caf_coffee", BASE_S + " + caf_soda + caf_tea + caf_energy")
binscreen("Soda caffeine (per 100 mg) -> prediabetes HbA1c", "prediab",
          "caf_soda", "caf_soda", BASE_S + " + BMXBMI + caf_coffee + caf_tea + caf_energy")
binscreen("Coffee caffeine (per 100 mg) -> prediabetes HbA1c", "prediab",
          "caf_coffee", "caf_coffee", BASE_S + " + BMXBMI + caf_soda + caf_tea + caf_energy")
SW = "C(sweet, Treatment('Black/low-sugar'))"
binscreen("Sweetened vs black coffee -> obesity", "obese", SW,
          f"{SW}[T.Sweetened (>=5 g sugar)]", BASE_S)
linscreen("Sweetened vs black coffee -> total daily kcal", "DR1TKCAL", SW,
          f"{SW}[T.Sweetened (>=5 g sugar)]", BASE_S)
linscreen("Sweetened vs black coffee -> sugar from OTHER foods (g)", "noncoffee_sugar", SW,
          f"{SW}[T.Sweetened (>=5 g sugar)]", BASE_S + " + DR1TKCAL")
scr = pd.DataFrame(rows)
for c in ["caf_soda", "caf_coffee"]:  # per-100-mg scaling for OR rows
    pass
log("\n=== COFFEE SIGNAL SCREEN (adjusted; design-based 95% CI) ===")
log("(per-mg rows are per 1 mg; see per-100 mg below)")
log(scr.round(3).to_string(index=False))
scr.to_csv(OUT / "screen.csv", index=False)

# per 100 mg ORs for the source comparison
log("\n=== SAME CAFFEINE, DIFFERENT SOURCE (OR per 100 mg) ===")
for y, extra in [("obese", ""), ("prediab", " + BMXBMI")]:
    m = des.glm(f"{y} ~ caf_coffee + caf_soda + caf_tea + caf_energy{BASE_S}{extra}", df,
                "logit")
    parts = []
    for k in ["caf_coffee", "caf_soda", "caf_tea", "caf_energy"]:
        o, lo, hi = m.ci(k, exp=True, scale=100)
        parts.append(f"{k[4:]}: {o:.2f} ({lo:.2f}-{hi:.2f}) p={m.p[k]:.2g}")
    log(f"{y:8s} " + " | ".join(parts))

log("\n=== SODA SPLIT: regular vs diet (OR per 100 mg caffeine, obesity) ===")
m = des.glm(f"obese ~ caf_coffee + caf_regsoda + caf_dietsoda + caf_tea + caf_energy{BASE_S}",
            df, "logit")
for k in ["caf_coffee", "caf_regsoda", "caf_dietsoda", "caf_tea", "caf_energy"]:
    o, lo, hi = m.ci(k, exp=True, scale=100)
    log(f"  {k[4:]:9s} OR {o:.2f} ({lo:.2f}-{hi:.2f}) p={m.p[k]:.2g}")
m = des.glm(f"obese ~ caf_coffee + caf_regsoda + caf_dietsoda + caf_tea + caf_energy"
            f"{BASE_S} + DR1TKCAL", df, "logit")
o, lo, hi = m.ci("caf_regsoda", exp=True, scale=100)
log(f"  regular soda, + total kcal adjusted: OR {o:.2f} ({lo:.2f}-{hi:.2f}) "
    f"p={m.p['caf_regsoda']:.2g}")
for s in ["regsoda", "dietsoda"]:
    log(f"  % adults drinking caffeinated {s}: "
        f"{100*des.mean((df[f'caf_{s}'] > 0).astype(float), df)[0]:.1f}%")

# dose-response prediabetes by cups (all levels)
log("\n=== DOSE-RESPONSE: cups/day -> prediabetes HbA1c (adj + BMI) ===")
m = des.glm(f"prediab ~ {CUPS}{BASE_S} + BMXBMI", df, "logit")
for lv in ["<1", "1-2", "2-4", ">4"]:
    k = f"{CUPS}[T.{lv}]"
    o, lo, hi = m.ci(k, exp=True)
    log(f"  {lv:>4s} cups: OR {o:.2f} ({lo:.2f}-{hi:.2f}) p={m.p[k]:.3g}")
m = des.glm(f"cur_smoke ~ {CUPS}{BASE}", df, "logit")
log("=== DOSE-RESPONSE: cups/day -> current smoking (adj) ===")
for lv in ["<1", "1-2", "2-4", ">4"]:
    k = f"{CUPS}[T.{lv}]"
    o, lo, hi = m.ci(k, exp=True)
    log(f"  {lv:>4s} cups: OR {o:.2f} ({lo:.2f}-{hi:.2f}) p={m.p[k]:.3g}")
LOG.close()
