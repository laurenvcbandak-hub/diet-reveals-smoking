"""Turn day-1 food records (DR1IFF) into per-person diet features.

Run as a script: builds 2017-March 2020 features, saves outputs/features.csv (one row per SEQN
in outputs/cohort.csv, with the cohort columns). `build_features()` is reused for other NHANES
cycles (external_validation.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd

CYCLE_2017 = {"iff": "data/P_DR1IFF.xpt", "fcd": "data/P_DRXFCD.xpt", "tot": "data/P_DR1TOT.xpt",
              "weight": "WTDRD1PP"}


def load(path):
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    # SAS transport stores zero as a tiny float (~5e-79); make it a real zero
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-10, 0)
    return df


def food_groups(desc, code):
    """Food-group masks: description keywords, scoped to the FNDDS code range they belong to."""
    prefix3 = code // 100_000       # e.g. 921 = coffee
    prefix2 = code // 1_000_000     # e.g. 63 = fruits
    is_diet = desc.str.contains(r"\bDIET\b|SUGAR[ -]FREE|LOW CALORIE|ZERO", regex=True)
    soda = desc.str.startswith("SOFT DRINK") | desc.str.startswith("WATER, TONIC")
    return {
        "soda_sugar": soda & ~is_diet,
        "soda_diet": soda & is_diet,
        "energy_drink": desc.str.startswith("ENERGY DRINK"),
        "coffee": (prefix3 == 921) & desc.str.contains("COFFEE"),
        "tea": (prefix3 == 923) & desc.str.contains(r"\bTEA\b", regex=True),
        "beer": (prefix3 == 931) & desc.str.contains("BEER"),
        "wine": (prefix3 == 934) & desc.str.contains("WINE"),
        "liquor": prefix3.isin([932, 933, 935]),          # liqueurs, cocktails, spirits
        "fruit": prefix2.isin([61, 62, 63]) & ~desc.str.contains("JUICE"),
        "fruit_juice": (prefix2 == 64) & desc.str.contains(r"JUICE|CIDER|NECTAR", regex=True),
        # vegetables excluding white potatoes (71), baby food (76), juices and condiments
        "vegetables": prefix2.isin([72, 73, 74, 75]) & ~desc.str.contains(
            r"JUICE|KETCHUP|MUSTARD|SAUCE|PICKLE|RELISH|CHUTNEY", regex=True),
    }


def build_features(cohort, files, verbose=True):
    """Diet features for everyone in `cohort`. Returns (cohort merged with features,
    list of feature columns, number of unmatched food codes)."""
    iff = load(files["iff"])
    iff = iff[iff["SEQN"].isin(cohort["SEQN"])]
    iff = iff[["SEQN", "DR1_020", "DR1IFDCD", "DR1IKCAL", "DR1IPROT", "DR1ICARB",
               "DR1ISUGR", "DR1IFIBE", "DR1ITFAT", "DR1IALCO", "DR1ICAFF"]].copy()

    fcd = load(files["fcd"])[["DRXFDCD", "DRXFCSD", "DRXFCLD"]]
    iff = iff.merge(fcd, left_on="DR1IFDCD", right_on="DRXFDCD", how="left")
    n_unmatched = int(iff["DRXFCSD"].isna().sum())
    iff["desc"] = iff["DRXFCSD"].fillna("").str.upper()
    groups = food_groups(iff["desc"], iff["DR1IFDCD"].astype("int64"))
    for g, mask in groups.items():
        iff[f"g_{g}"] = mask.astype(int)
        iff[f"k_{g}"] = iff["DR1IKCAL"].where(mask, 0)

    if verbose:
        print(f"Food records for cohort: {len(iff):,} (unmatched food codes: {n_unmatched})\n")
        print("=" * 70)
        print("Top 15 food descriptions per group (number of records)")
        print("=" * 70)
        for g, mask in groups.items():
            top = iff.loc[mask, "DRXFCSD"].value_counts().head(15)
            print(f"\n[{g}]  {mask.sum():,} records, {iff.loc[mask, 'DRXFCSD'].nunique()} distinct foods")
            for d, n in top.items():
                print(f"  {n:>6,}  {d}")
        print()

    # Per-person totals
    agg = iff.groupby("SEQN").agg(
        kcal=("DR1IKCAL", "sum"),
        sugar_g=("DR1ISUGR", "sum"),
        fiber_g=("DR1IFIBE", "sum"),
        alcohol_g=("DR1IALCO", "sum"),
        caffeine_mg=("DR1ICAFF", "sum"),
        protein_g=("DR1IPROT", "sum"),
        fat_g=("DR1ITFAT", "sum"),
        carb_g=("DR1ICARB", "sum"),
        n_food_codes=("DR1IFDCD", "nunique"),
    )
    kcal = agg["kcal"].replace(0, np.nan)
    agg["pct_kcal_sugar"] = 100 * agg["sugar_g"] * 4 / kcal
    agg["pct_kcal_alcohol"] = 100 * agg["alcohol_g"] * 7 / kcal

    grp = iff.groupby("SEQN")[[f"g_{g}" for g in groups] + [f"k_{g}" for g in groups]].sum()
    for g in groups:
        agg[f"any_{g}"] = (grp[f"g_{g}"] > 0).astype(int)
        agg[f"pct_kcal_{g}"] = 100 * grp[f"k_{g}"] / kcal

    # Timing: only items with energy or caffeine
    t = iff[(iff["DR1IKCAL"] > 0) | (iff["DR1ICAFF"] > 0)].copy()
    t["hour"] = t["DR1_020"] / 3600
    timing = t.groupby("SEQN").agg(first_eat_h=("hour", "min"), last_eat_h=("hour", "max"))
    timing["eating_window_h"] = timing["last_eat_h"] - timing["first_eat_h"]
    timing["first_caffeine_h"] = t[t["DR1ICAFF"] > 0].groupby("SEQN")["hour"].min()
    late_kcal = t[t["hour"] >= 20].groupby("SEQN")["DR1IKCAL"].sum()
    timing["pct_kcal_after_20h"] = (100 * late_kcal.reindex(timing.index, fill_value=0)
                                    / kcal.reindex(timing.index))

    # Plain water from daily totals
    tot = load(files["tot"])[["SEQN", "DR1_320Z"]].set_index("SEQN")
    water = tot.rename(columns={"DR1_320Z": "plain_water_g"})
    water["any_plain_water"] = (water["plain_water_g"] > 0).astype(float).where(
        water["plain_water_g"].notna())

    features = agg.join(timing).join(water, how="left")
    features.index = features.index.astype(int)
    out = cohort.merge(features, left_on="SEQN", right_index=True, how="left")
    return out, list(features.columns), n_unmatched


def wmean_sd(x, w):
    ok = x.notna()
    x, w = x[ok], w[ok]
    m = np.average(x, weights=w)
    return m, np.sqrt(np.average((x - m) ** 2, weights=w))


if __name__ == "__main__":
    cohort = pd.read_csv("outputs/cohort.csv")
    out, feature_cols, _ = build_features(cohort, CYCLE_2017)

    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/features.csv", index=False)
    print(f"Saved {len(out):,} rows x {out.shape[1]} columns ({len(feature_cols)} features) "
          f"to outputs/features.csv")
    print(f"Cohort members with no food records: {out['kcal'].isna().sum()}\n")

    # Current vs never smokers: survey-weighted means
    wcol = CYCLE_2017["weight"]
    cur = out[out["smoking_status"] == "current"]
    nev = out[out["smoking_status"] == "never"]
    rows = []
    for c in feature_cols:
        m1, s1 = wmean_sd(cur[c], cur[wcol])
        m0, s0 = wmean_sd(nev[c], nev[wcol])
        pooled = np.sqrt((s1 ** 2 + s0 ** 2) / 2)
        rows.append((c, m1, m0, m1 - m0, (m1 - m0) / pooled if pooled > 0 else np.nan))
    cmp = pd.DataFrame(rows, columns=["feature", "current", "never", "diff", "std_diff"])
    cmp = cmp.reindex(cmp["std_diff"].abs().sort_values(ascending=False).index)

    print("Survey-weighted means, current (n={:,}) vs never (n={:,}) smokers".format(len(cur), len(nev)))
    print("Sorted by |standardized difference| (diff / pooled weighted SD); any_* are proportions")
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 120):
        print(cmp.to_string(index=False))
