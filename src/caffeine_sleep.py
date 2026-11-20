"""
EDA: Is late-day caffeine linked to shorter sleep, daytime sleepiness and sleep trouble
in US adults? Is there a cutoff time after which caffeine matters?
NHANES 2017-March 2020 pre-pandemic.

Run from project root:  python src/caffeine_sleep.py
Outputs: outputs/caffeine/{figures,tables}/, outputs/caffeine/log.txt
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
OUT = Path("outputs/caffeine")
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
    df[num] = df[num].mask(df[num].abs() < 1e-70, 0.0)
    return df


def hhmm(s):
    try:
        h, m = str(s).split(":")
        return int(h) + int(m) / 60
    except Exception:
        return np.nan


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
files = ["P_DEMO", "P_SLQ", "P_DR1TOT", "P_DR1IFF", "P_BMX", "P_SMQ", "P_ALQ", "P_PAQ"]
raw = {f: read(f) for f in files}
des = Design(raw["P_DEMO"])
log("=== 1. FILES ===")
for f, d in raw.items():
    log(f"{f:9s} rows={len(d):7d}")
log(f"Design: {des.n_strata} strata, design df = {des.ddf}")

# --------------------------------------------------------------------------
# 2. Caffeine exposures from the food-by-food recall
# --------------------------------------------------------------------------
iff = raw["P_DR1IFF"]
iff = iff[iff.DR1DRSTZ == 1].copy()
iff["t"] = iff.DR1_020 / 3600
iff["caf"] = iff.DR1ICAFF.fillna(0)
# 00:00-03:59 items belong to the previous night, not the evening before this night's sleep
day = iff[iff.t >= 4]

per = pd.DataFrame(index=iff.SEQN.unique())
per["caf_total"] = iff.groupby("SEQN").caf.sum()
CUTS = [10, 12, 14, 16, 18, 20]
for c in CUTS:
    per[f"caf_after{c}"] = day[day.t >= c].groupby("SEQN").caf.sum()
per["caf_before12"] = day[day.t < 12].groupby("SEQN").caf.sum()
per["last_caf"] = day[day.caf >= 10].groupby("SEQN").t.max()  # >=10 mg (ignore traces)
per = per.fillna({c: 0 for c in per.columns if c != "last_caf"})
per.index.name = "SEQN"

# hourly caffeine clock (for figure)
hourly = day.assign(h=day.t.astype(int))  # item file already carries WTDRD1PP

# --------------------------------------------------------------------------
# 3. Merge + derive outcomes / covariates
# --------------------------------------------------------------------------
df = raw["P_DEMO"].merge(raw["P_DR1TOT"][["SEQN", "DR1DRSTZ", "DR1DAY", "DR1TKCAL",
                                          "DR1TALCO", "WTDRD1PP"]], on="SEQN")
for f in ["P_SLQ", "P_BMX", "P_SMQ", "P_ALQ", "P_PAQ"]:
    df = df.merge(raw[f], on="SEQN", how="left")
df = df.merge(per, left_on="SEQN", right_index=True, how="left")

weeknight = df.DR1DAY.between(1, 5)  # Sun-Thu intake day -> work/school night follows
df["bed"] = np.where(weeknight, df.SLQ300.map(hhmm), df.SLQ320.map(hhmm))
df["bed_c"] = np.where(df.bed < 12, df.bed + 24, df.bed)
df["sleep_h"] = np.where(weeknight, df.SLD012, df.SLD013)
df["sleep_h_wd"] = df.SLD012
df["short_sleep"] = (df.sleep_h < 7).astype(float).where(df.sleep_h.notna())
df["sleepy"] = (df.SLQ120 >= 3).astype(float).where(df.SLQ120.between(0, 4))
df["trouble"] = (df.SLQ050 == 1).astype(float).where(df.SLQ050.isin([1, 2]))
df["hrs_caf_before_bed"] = df.bed_c - df.last_caf

df["female"] = (df.RIAGENDR == 2).astype(int)
df["race"] = df.RIDRETH3.map({1: "MexAm", 2: "OthHisp", 3: "NHWhite", 4: "NHBlack",
                              6: "NHAsian", 7: "Other"})
df["educ"] = df.DMDEDUC2.map({1: "<=HS", 2: "<=HS", 3: "<=HS", 4: "SomeColl", 5: "CollGrad"})
df["smoke"] = np.select([df.SMQ020 == 2, (df.SMQ020 == 1) & (df.SMQ040 == 3),
                         (df.SMQ020 == 1) & df.SMQ040.isin([1, 2])],
                        ["Never", "Former", "Current"], default=None)
FREQ_WK = {0: 0, 1: 7, 2: 5.5, 3: 3.5, 4: 2, 5: 1, 6: 2.5 / 4.33, 7: 1 / 4.33,
           8: 9 / 52, 9: 4.5 / 52, 10: 1.5 / 52}
fw = np.where(df.ALQ111 == 2, 0, df.ALQ121.map(FREQ_WK))
dpw = np.where(fw == 0, 0, fw * df.ALQ130.where(df.ALQ130 < 777))
cut = np.where(df.female == 1, 7, 14)
df["alcohol"] = np.select([dpw == 0, dpw <= cut, dpw > cut], ["None", "Moderate", "Heavy"],
                          default=None)
df.loc[pd.isna(dpw), "alcohol"] = None


def mins(days, m):
    d = df[days].where(df[days] <= 7)
    return (d * df[m].where(df[m] < 7777)).fillna(0)


df["mvpa"] = (2 * mins("PAQ610", "PAD615") + mins("PAQ625", "PAD630") + mins("PAQ640", "PAD645")
              + 2 * mins("PAQ655", "PAD660") + mins("PAQ670", "PAD675"))
df.loc[df.PAQ605.isna() & df.PAQ650.isna(), "mvpa"] = np.nan
df["pa"] = pd.cut(df.mvpa, [-1, 0, 149.99, 1e9],
                  labels=["Inactive", "Insufficient", "Active"]).astype(object)
df["w"] = df.WTDRD1PP

# --------------------------------------------------------------------------
# 4. Cohort
# --------------------------------------------------------------------------
funnel = [("All participants", len(raw["P_DEMO"])), ("With day-1 diet record", len(df))]


def step(label, mask):
    global df
    df = df[mask]
    funnel.append((label, len(df)))


step("Age >= 20", df.RIDAGEYR >= 20)
step("Not pregnant", df.RIDEXPRG != 1)
step("Reliable day-1 recall, diet weight > 0", (df.DR1DRSTZ == 1) & (df.w > 0))
step("Sleep duration + bedtime reported", df.sleep_h.notna() & df.bed.notna())
step("Bedtime 18:00-06:00 (excl. likely night-shift workers)", df.bed_c.between(18, 30))
COVS = ["RIDAGEYR", "female", "race", "educ", "INDFMPIR", "BMXBMI", "smoke", "alcohol", "pa"]
ALL = df.copy()
step("Complete covariates", df[COVS].notna().all(axis=1))
log("\n=== 2. COHORT FUNNEL ===")
for lbl, n in funnel:
    log(f"{n:7d}  {lbl}")
pd.DataFrame(funnel, columns=["step", "n"]).to_csv(TAB / "cohort_funnel.csv", index=False)

# caffeine within X hours before bedtime (needs item-level merge with bedtime)
it = day[["SEQN", "t", "caf"]].merge(df[["SEQN", "bed_c"]], on="SEQN")
it["hb"] = it.bed_c - it.t
for h in [2, 4, 6, 8, 10, 12]:
    s = it[(it.hb >= 0) & (it.hb < h)].groupby("SEQN").caf.sum()
    df[f"caf_within{h}h"] = df.SEQN.map(s).fillna(0)
df.to_csv(OUT / "analytic_cohort.csv", index=False)

# --------------------------------------------------------------------------
# 5. Descriptives
# --------------------------------------------------------------------------
log("\n=== 3. CAFFEINE + SLEEP DESCRIPTIVES (weighted) ===")
consumers = df.caf_total >= 10
log(f"N={len(df)};  % with >=10 mg caffeine: {100*des.mean(consumers.astype(float), df)[0]:.1f}%")
for v, lab in [("caf_total", "Total caffeine mg/d"), ("caf_after14", "Caffeine after 14:00 mg"),
               ("caf_after16", "Caffeine after 16:00 mg"), ("sleep_h", "Sleep h (matching night)"),
               ("bed_c", "Bedtime (24=midnight)")]:
    est, lo, hi = des.mean(df[v], df)
    log(f"  {lab:28s} mean {est:6.1f} ({lo:.1f}-{hi:.1f});  median {df[v].median():.1f};  "
        f"p90 {df[v].quantile(.9):.1f}")
c = df[consumers]
log(f"Among consumers: median last-caffeine clock time {c.last_caf.median():.1f} h; "
    f"median hours caffeine->bed {c.hrs_caf_before_bed.median():.1f}")
for v, lab in [("short_sleep", "% short sleep (<7 h)"), ("sleepy", "% often sleepy in day"),
               ("trouble", "% told doctor trouble sleeping")]:
    log(f"  {lab:32s} {100*des.mean(df[v], df)[0]:.1f}%")
log(f"  % with any caffeine within 6 h of bed: "
    f"{100*des.mean((df.caf_within6h >= 10).astype(float), df)[0]:.1f}%; "
    f"within 8.8 h-ish (8 h) {100*des.mean((df.caf_within8h >= 10).astype(float), df)[0]:.1f}%")

# Fig 1: caffeine clock + distributions
hw = hourly[hourly.SEQN.isin(df.SEQN)]
clock = hw.groupby("h").apply(lambda g: (g.caf * g.WTDRD1PP).sum()) / df.w.sum()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].bar(clock.index, clock.values, color="#8C564B")
ax[0].set(title="When Americans drink caffeine", xlabel="Clock hour",
          ylabel="Mean mg per adult (weighted)")
ax[1].hist(c.last_caf, bins=np.arange(4, 24.5, .5), color="#C44E52", edgecolor="white")
ax[1].set(title="Time of last caffeine (consumers)", xlabel="Clock hour", ylabel="People")
ax[2].hist(c.hrs_caf_before_bed.clip(-2, 20), bins=np.arange(-2, 20.5, .5), color="#4C72B0",
           edgecolor="white")
ax[2].axvline(6, color="k", ls="--", lw=1, label="6 h (Drake 2013)")
ax[2].axvline(8.8, color="crimson", ls="--", lw=1, label="8.8 h (Gardiner 2023)")
ax[2].set(title="Hours from last caffeine to bedtime", xlabel="Hours", ylabel="People")
ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "01_caffeine_timing.png", dpi=130); plt.close(fig)

# --------------------------------------------------------------------------
# 6. Unadjusted: outcomes by afternoon caffeine dose
# --------------------------------------------------------------------------
df["caf14_cat"] = pd.cut(df.caf_after14, [-1, 9.99, 95, 200, 1e5],
                         labels=["None", "<=95 mg", "96-200 mg", ">200 mg"]).astype(str)
rows = []
for lvl in ["None", "<=95 mg", "96-200 mg", ">200 mg"]:
    sub = df.assign(w=df.w.where(df.caf14_cat == lvl, 0))
    m = df.caf14_cat == lvl
    r = {"group": lvl, "n": int(m.sum())}
    r["sleep_h"] = des.mean(df.sleep_h.where(m), sub)[0]
    for v in ["short_sleep", "sleepy", "trouble"]:
        r[v] = 100 * des.mean(df[v].where(m), sub)[0]
    r["total_caf"] = des.mean(df.caf_total.where(m), sub)[0]
    r["age"] = des.mean(df.RIDAGEYR.where(m), sub)[0]
    r["%current_smoker"] = 100 * des.mean((df.smoke == "Current").astype(float).where(m), sub)[0]
    rows.append(r)
t2 = pd.DataFrame(rows).set_index("group")
log("\n=== 4. OUTCOMES BY CAFFEINE AFTER 14:00 (unadjusted, weighted) ===")
log(t2.round(2).to_string())
t2.to_csv(TAB / "outcomes_by_afternoon_caffeine.csv")

# --------------------------------------------------------------------------
# 7. Adjusted models
# --------------------------------------------------------------------------
ADJ = (" + RIDAGEYR + I(RIDAGEYR**2) + female + C(race) + C(educ) + INDFMPIR + BMXBMI"
       " + C(smoke) + C(alcohol) + C(pa) + DR1TKCAL")
OUTC = [("sleep_h", "gaussian", "Sleep duration (min)", 60),
        ("short_sleep", "logit", "Short sleep <7h (OR)", 1),
        ("sleepy", "logit", "Often sleepy (OR)", 1),
        ("trouble", "logit", "Trouble sleeping (OR)", 1)]


def eff(m, term, fam, scale):
    """Effect per 100 mg: minutes of sleep (gaussian) or OR (logit)."""
    if fam == "gaussian":
        return m.ci(term, scale=100 * scale)
    return m.ci(term, exp=True, scale=100)


log("\n=== 5. ADJUSTED: per 100 mg caffeine (design-based 95% CI) ===")
res = []
for y, fam, lab, sc in OUTC:
    d = df.dropna(subset=[y])
    m_tot = des.glm(f"{y} ~ caf_total{ADJ}", d, fam)
    m_split = des.glm(f"{y} ~ caf_before12 + I(caf_total - caf_before12 - caf_after16)"
                      f" + caf_after16{ADJ}", d, fam)
    e_tot = eff(m_tot, "caf_total", fam, sc)
    e_am = eff(m_split, "caf_before12", fam, sc)
    e_pm = eff(m_split, "caf_after16", fam, sc)
    log(f"{lab:24s} total: {e_tot[0]:+.2f} ({e_tot[1]:+.2f},{e_tot[2]:+.2f}) p={m_tot.p['caf_total']:.3g}"
        f" | before 12:00: {e_am[0]:+.2f} ({e_am[1]:+.2f},{e_am[2]:+.2f}) "
        f"p={m_split.p['caf_before12']:.3g} | after 16:00: {e_pm[0]:+.2f} "
        f"({e_pm[1]:+.2f},{e_pm[2]:+.2f}) p={m_split.p['caf_after16']:.3g}")
    res.append((lab, "total", *e_tot, m_tot.p["caf_total"], m_tot.n))
    res.append((lab, "before 12:00", *e_am, m_split.p["caf_before12"], m_split.n))
    res.append((lab, "after 16:00", *e_pm, m_split.p["caf_after16"], m_split.n))
pd.DataFrame(res, columns=["outcome", "exposure", "est", "lo", "hi", "p", "n"]).to_csv(
    TAB / "adjusted_am_vs_pm.csv", index=False)

# --------------------------------------------------------------------------
# 8. Cutoff scans: (a) clock time, (b) hours before bedtime
#    Each model: caffeine after cutoff + caffeine before cutoff (=> holds total fixed)
# --------------------------------------------------------------------------
log("\n=== 6. CUTOFF SCAN: effect per 100 mg consumed AFTER clock time X (total held fixed) ===")
scan = []
for y, fam, lab, sc in OUTC:
    d = df.dropna(subset=[y])
    for c in CUTS:
        m = des.glm(f"{y} ~ caf_after{c} + I(caf_total - caf_after{c}){ADJ}", d, fam)
        e = eff(m, f"caf_after{c}", fam, sc)
        scan.append(("clock", lab, c, *e, m.p[f"caf_after{c}"]))
    for h in [2, 4, 6, 8, 10, 12]:
        m = des.glm(f"{y} ~ caf_within{h}h + I(caf_total - caf_within{h}h){ADJ}", d, fam)
        e = eff(m, f"caf_within{h}h", fam, sc)
        scan.append(("bedtime", lab, h, *e, m.p[f"caf_within{h}h"]))
        # same, also adjusting for bedtime (early sleepers have more caffeine 'near bed'
        # for the same clock time, and early sleepers sleep longer)
        m = des.glm(f"{y} ~ caf_within{h}h + I(caf_total - caf_within{h}h) + bed_c{ADJ}",
                    d, fam)
        e = eff(m, f"caf_within{h}h", fam, sc)
        scan.append(("bedtime+adj", lab, h, *e, m.p[f"caf_within{h}h"]))
scan = pd.DataFrame(scan, columns=["kind", "outcome", "cut", "est", "lo", "hi", "p"])
scan.to_csv(TAB / "cutoff_scan.csv", index=False)
for (k, o), g in scan.groupby(["kind", "outcome"], sort=False):
    unit = {"clock": "after h", "bedtime": "within h of bed",
            "bedtime+adj": "within h of bed, +bedtime adj"}[k]
    log(f"{o:24s} [{unit}] " + "  ".join(f"{int(r.cut)}:{r.est:+.2f}{'*' if r.p < .05 else ''}"
                                         for r in g.itertuples()))
log("  (* p<0.05; sleep in minutes per 100 mg, others OR per 100 mg)")

fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
for j, (y, fam, lab, sc) in enumerate(OUTC):
    for i, k in enumerate(["clock", "bedtime"]):
        a = ax[i, j]
        kinds = [("clock", "#4C72B0", "")] if k == "clock" else [
            ("bedtime", "#C44E52", "not adj. for bedtime"),
            ("bedtime+adj", "#55A868", "adj. for bedtime")]
        for off, (kk, col, lbl) in enumerate(kinds):
            g = scan[(scan.kind == kk) & (scan.outcome == lab)]
            a.errorbar(g.cut + 0.25 * off, g.est, yerr=[g.est - g.lo, g.hi - g.est], fmt="o-",
                       capsize=3, color=col, label=lbl or None)
        if k == "bedtime" and j == 0:
            a.legend(fontsize=8)
        a.axhline(0 if fam == "gaussian" else 1, color="k", lw=.8)
        a.set_title(lab, fontsize=10)
        a.set_xlabel("Caffeine consumed after this clock hour" if k == "clock"
                     else "Caffeine within this many hours of bedtime")
        if fam == "logit":
            a.set_yscale("log")
fig.suptitle("Is caffeine later in the day worse? Effect per 100 mg, holding total caffeine fixed "
             "(adjusted, 95% CI)")
fig.tight_layout(); fig.savefig(FIG / "02_cutoff_scan.png", dpi=130); plt.close(fig)

# --------------------------------------------------------------------------
# 9. Reverse-causation / confounding check: non-consumers vs morning-only vs late
# --------------------------------------------------------------------------
log("\n=== 7. PATTERN GROUPS (adjusted, vs no caffeine) ===")
df["pattern"] = np.select(
    [df.caf_total < 10, (df.caf_total >= 10) & (df.caf_after14 < 10), df.caf_after14 >= 10],
    ["None", "Morning/midday only", "Some after 14:00"], default="")
df["pattern"] = df.pattern.replace("", np.nan)  # no food records at all
pat = []
for y, fam, lab, sc in OUTC:
    d = df.dropna(subset=[y])
    m = des.glm(f"{y} ~ C(pattern, Treatment('None')){ADJ}", d, fam)
    parts = []
    for lv in ["Morning/midday only", "Some after 14:00"]:
        k = f"C(pattern, Treatment('None'))[T.{lv}]"
        e = m.ci(k, scale=sc) if fam == "gaussian" else m.ci(k, exp=True)
        parts.append(f"{lv}: {e[0]:+.2f} ({e[1]:+.2f},{e[2]:+.2f}) p={m.p[k]:.3g}")
        pat.append((lab, lv, *e, m.p[k]))
    log(f"{lab:24s} " + " | ".join(parts))
pd.DataFrame(pat, columns=["outcome", "pattern", "est", "lo", "hi", "p"]).to_csv(
    TAB / "pattern_groups.csv", index=False)

log("\n=== 7b. THRESHOLD CHECK: afternoon dose groups vs none after 14:00 (adjusted, "
    "+ total caffeine) ===")
thr = []
for y, fam, lab, sc in OUTC:
    d = df.dropna(subset=[y])
    m = des.glm(f"{y} ~ C(caf14_cat, Treatment('None')) + caf_total{ADJ}", d, fam)
    parts = []
    for lv in ["<=95 mg", "96-200 mg", ">200 mg"]:
        k = f"C(caf14_cat, Treatment('None'))[T.{lv}]"
        e = m.ci(k, scale=sc) if fam == "gaussian" else m.ci(k, exp=True)
        parts.append(f"{lv}: {e[0]:+.2f} ({e[1]:+.2f},{e[2]:+.2f}) p={m.p[k]:.3g}")
        thr.append((lab, lv, *e, m.p[k]))
    log(f"{lab:24s} " + " | ".join(parts))
pd.DataFrame(thr, columns=["outcome", "afternoon_dose", "est", "lo", "hi", "p"]).to_csv(
    TAB / "afternoon_dose_groups.csv", index=False)
log("  n per pattern: " + str(df.pattern.value_counts().to_dict()))

# --------------------------------------------------------------------------
# 10. Correlates of late caffeine (who are the late drinkers?)
# --------------------------------------------------------------------------
log("\n=== 8. WHO DRINKS CAFFEINE AFTER 14:00? (weighted means) ===")
prof = {}
for lv in ["None", "Morning/midday only", "Some after 14:00"]:
    m = df.pattern == lv
    sub = df.assign(w=df.w.where(m, 0))
    prof[lv] = {
        "n": int(m.sum()),
        "age": des.mean(df.RIDAGEYR.where(m), sub)[0],
        "% female": 100 * des.mean(df.female.astype(float).where(m), sub)[0],
        "% current smoker": 100 * des.mean((df.smoke == "Current").astype(float).where(m), sub)[0],
        "BMI": des.mean(df.BMXBMI.where(m), sub)[0],
        "bedtime": des.mean(df.bed_c.where(m), sub)[0],
        "total caffeine mg": des.mean(df.caf_total.where(m), sub)[0],
        "sleep h": des.mean(df.sleep_h.where(m), sub)[0],
    }
prof = pd.DataFrame(prof)
log(prof.round(1).to_string())
prof.to_csv(TAB / "profile_by_pattern.csv")

LOG.close()
print("\nDone. See outputs/caffeine/")
