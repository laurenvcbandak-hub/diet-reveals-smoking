# diet-reveals-smoking

Can one day of food records identify current smokers? This project uses machine learning on US national survey data to test whether what people eat in a single day reveals whether they smoke, and how much diet adds beyond basic demographics.

## Research question

In NHANES 2017–March 2020, does a machine-learning model that adds one day of 24-hour dietary recall features predict current cigarette smoking in US adults better than a baseline model using demographics alone (age, sex, ethnicity, education, income)?

**Answer: yes.** Adding diet raised the AUC from 0.744 to 0.808 on unseen test data (ΔAUC +0.064, 95% CI +0.037 to +0.088), and the same gain was found on a separate NHANES cycle (2015–2016).

## Data

- **Source:** National Health and Nutrition Examination Survey (NHANES), CDC / NCHS, cycle 2017–March 2020 Pre-Pandemic ("P_" files): https://wwwn.cdc.gov/nchs/nhanes
- **Key:** `SEQN` (participant ID) links all files. In `P_DR1IFF` each row is one food item (`SEQN` + `DR1ILINE`).
- **Sample:** 7,703 adults aged 20+ with a reliable day-1 food recall: 1,405 current, 1,862 former and 4,436 never smokers (17.0% current, survey-weighted). External validation: 5,008 adults from NHANES 2015–2016.

| File | What it gives |
|---|---|
| `P_DEMO` | Age, sex, ethnicity, education, income, survey design variables |
| `P_SMQ` | Smoking status, cigarettes per day, time since quitting |
| `P_DR1IFF` | Every food eaten on day 1, with time, food code and nutrients |
| `P_DR1TOT` | Daily nutrient totals, recall quality flag, dietary weight |
| `P_DRXFCD` | Food code descriptions (to identify soda, beer, coffee, fruit…) |
| `P_BMX` | BMI and waist circumference |
| `P_COT` | Serum cotinine, to check self-reported smoking |
| `data/2015/*_I` | The same files for NHANES 2015–2016 (external validation) |

The data is **not** stored in this repository. The files are public and free: download the `.xpt` files from the NHANES website into a local `data/` folder (ignored by Git).

## Related work

**Lin et al. (2024), *Nutrients* 16(13):2035.** Using NHANES 2017–2018 (3,199 adults), smokers ate frozen meals and pizza more often than non-smokers (adjusted rate ratio 1.7) and fewer had heard of MyPlate (21% vs 30%). Diet was measured with only three questionnaire items, and former smokers were not studied separately.

**Luo & Tseng (2024), *BMC Public Health* 24:1212.** Using NHANES 2005–2018 (31,569 adults), current smokers had lower diet quality (HEI 49.2) than former (54.0) and never smokers (53.3). Heavy smokers scored lower than light smokers, and diet quality reached never-smoker levels 5–10 years after quitting. Diet was reduced to one overall score and no prediction model was built.

**Gap:** Neither study tested whether diet can *identify* smokers, or how much diet adds beyond demographics. Both also lose specific habits such as caffeine, soda, alcohol and meal timing.

## Approach

1. **Cohort:** adults 20+ with a reliable day-1 recall and known smoking status. Main outcome: current vs never smokers.
2. **Features:** 40 diet features: nutrients, food groups (soda, beer, coffee, fruit, water…), food variety and meal timing (first caffeine, late-evening eating).
3. **Baseline:** logistic regression on demographics only.
4. **Models:** diet-only and demographics (+ body size) + diet, using regularized logistic regression and gradient boosting.
5. **Secondary analyses:** a diet "smoker score" applied to former smokers by time since quitting, and to current smokers by cigarettes per day; performance by sex, age and ethnicity.

**Validation:** stratified 5-fold cross-validation, a 20% held-out test set, bootstrap 95% confidence intervals, external validation on NHANES 2015–2016, a serum cotinine check and survey-weighted AUC.

## Results

AUC for current vs never smokers (0.5 = coin flip, 1 = perfect):

| Model | 5-fold CV (2017–2020) | Held-out test (2017–2020) | External (2015–2016) |
|---|---|---|---|
| Demographics only | 0.748 | 0.744 | 0.740 |
| Diet only | 0.770 | – | – |
| Demographics + body + diet | 0.821 | 0.808 | 0.804 |
| **Gain from diet (ΔAUC, 95% CI)** | +0.065 [+0.052, +0.078] | +0.064 [+0.037, +0.088] | +0.064 [+0.049, +0.082] |

**The smoker diet:** compared with never smokers, current smokers ate fewer different foods (11.8 vs 14.1), took in almost twice the caffeine (257 vs 134 mg), had their first caffeine over an hour earlier, drank more sugary soda and beer, and ate less fruit, vegetables and fiber.

**Dose:** the diet smoker score rose with cigarettes per day (1–5: 0.32; >20: 0.60; p < 0.001).

**Quitting:** the score was lower the longer ago people quit (< 1 year: 0.30; > 20 years: 0.21; never smokers: 0.20; p < 0.001).

**Fairness:** diet-only AUC was highest for Non-Hispanic White adults (0.80) and lowest for Non-Hispanic Black and Asian adults (0.70–0.71).

**Robustness:** the gain from diet held after removing participants whose serum cotinine disagreed with their answer (ΔAUC +0.065 at 3 ng/mL) and with survey weights (+0.075).

## Repository structure

```
diet-reveals-smoking/
├── data/          # NHANES .xpt files (not in Git)
├── literature/    # paper notes
├── outputs/       # generated tables and figures (not in Git)
├── src/           # analysis scripts
└── README.md
```

| Script | What it does |
|---|---|
| `src/check_files.py` | Checks that all data files load |
| `src/build_cohort.py` | Selects participants and defines smoking status |
| `src/build_features.py` | Builds the 40 diet features |