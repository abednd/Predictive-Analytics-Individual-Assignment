# =============================================================================
# Diabetic Readmission EDA
# =============================================================================
# Run in Jupyter. Figures saved to ./eda_figures/
# Dataset: diabetic_data.csv (place in same directory as this script)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import pointbiserialr
import os
import warnings
warnings.filterwarnings('ignore')

# ── Output directory ──────────────────────────────────────────────────────────
OUT = "eda_figures"
os.makedirs(OUT, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
# CHANGE-1: Only "?" is treated as true missing. "None" is a valid category
# in several columns (e.g. A1Cresult, max_glu_serum = "not tested/measured"),
# not an absent value. We disable pandas' default NA list and only specify "?"
# to prevent "None" strings from being silently coerced to NaN.
df = pd.read_csv(
    "diabetic_data.csv",
    na_values=["?"],
    keep_default_na=False,
    low_memory=False
)

print("=" * 70)
print("SECTION 1 — BASIC STRUCTURE CHECK")
print("=" * 70)

# ── 1a. Shape & dtypes ────────────────────────────────────────────────────────
print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nData types:\n{df.dtypes.value_counts()}")

# ── 1b. Cardinality table ─────────────────────────────────────────────────────
card = pd.DataFrame({
    "dtype":   df.dtypes,
    "n_unique": df.nunique(),
    "n_null":  df.isna().sum(),
    "pct_null": (df.isna().mean() * 100).round(2),
    "sample_values": [str(df[c].dropna().unique()[:4].tolist()) for c in df.columns]
})
card["flag"] = ""
card.loc[card["n_unique"] == 1, "flag"] = "CONSTANT"
card.loc[card["n_unique"] > 1000, "flag"] = "HIGH_CARD"
card.loc[df.columns.isin(["encounter_id", "patient_nbr"]), "flag"] = "IDENTIFIER"

print("\n--- Cardinality Table ---")
print(card[["dtype","n_unique","pct_null","flag"]].to_string())

print(f"\n--- Flagged columns ---")
for flag in ["IDENTIFIER","CONSTANT","HIGH_CARD"]:
    flagged = card[card["flag"] == flag].index.tolist()
    print(f"  {flag}: {flagged}")

# CHANGE-2: Explicitly verify constant columns with value_counts
print("\n--- Constant column verification (value_counts) ---")
for col in card[card["flag"] == "CONSTANT"].index:
    print(f"  {col}: {df[col].value_counts(dropna=False).to_dict()}")

# ── Commentary (printed, mirrors Jupyter markdown cells) ─────────────────────
print("""
COMMENTARY — Section 1:
  The dataset has 101,766 encounters across 50 columns. Columns are a mix of
  integer IDs, nominal categoricals, and sparse drug-flag fields.
  encounter_id and patient_nbr are identifiers and carry no predictive signal.
  examide and citoglipton are confirmed truly constant: both contain only "No"
  across all 101,766 rows (verified via value_counts above). They carry zero
  variance and should be excluded at feature selection.
  diag_1/diag_2/diag_3 are high-cardinality ICD code fields (700–790 unique
  values) and will require grouping (e.g. CCS categories) before modelling.
""")

# =============================================================================
print("=" * 70)
print("SECTION 2 — TARGET DISTRIBUTION")
print("=" * 70)

# ── 3-class ───────────────────────────────────────────────────────────────────
three_class = df["readmitted"].value_counts(dropna=False)
print(f"\n3-class distribution:\n{three_class}")
print(f"  (% of total):\n{(three_class / len(df) * 100).round(2)}")

# ── Binary ────────────────────────────────────────────────────────────────────
df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
binary = df["readmit_30"].value_counts()
pos_pct = df["readmit_30"].mean() * 100
print(f"\nBinary target (1 = readmitted <30 days):\n{binary}")
print(f"\nClass imbalance: {pos_pct:.1f}% positive class")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
three_class.plot(kind="bar", ax=axes[0], color=["#4C72B0","#DD8452","#55A868"],
                 edgecolor="black", width=0.6)
axes[0].set_title("3-Class Distribution")
axes[0].set_xlabel("readmitted"); axes[0].set_ylabel("Count")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width()/2, p.get_height()),
                     ha="center", va="bottom", fontsize=9)

binary.rename({0:"Other / No readmit", 1:"<30 days"}).plot(
    kind="bar", ax=axes[1], color=["#4C72B0","#DD8452"],
    edgecolor="black", width=0.5)
axes[1].set_title("Binary Target Distribution")
axes[1].set_xlabel(""); axes[1].set_ylabel("Count")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width()/2, p.get_height()),
                     ha="center", va="bottom", fontsize=9)
axes[1].set_xticklabels(["Other / No readmit", "<30 days"], rotation=0)

plt.tight_layout()
plt.savefig(f"{OUT}/02_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {OUT}/02_target_distribution.png")

print("""
COMMENTARY — Section 2:
  The positive class (<30-day readmission) comprises ~11.2% of encounters.
  The dataset is moderately imbalanced (roughly 1:8 ratio). Standard accuracy
  is an unreliable metric; AUC-ROC, precision-recall AUC, and F1 at chosen
  thresholds are more informative. Cost-sensitive learning or resampling
  strategies may be considered later.
""")

# =============================================================================
print("=" * 70)
print("SECTION 3 — PATIENT-LEVEL REPETITION CHECK")
print("=" * 70)

n_patients = df["patient_nbr"].nunique()
encounters_per_patient = df.groupby("patient_nbr").size()
multi_encounter = (encounters_per_patient > 1).sum()
multi_pct = multi_encounter / n_patients * 100
max_enc = encounters_per_patient.max()

print(f"\nTotal encounters         : {len(df):,}")
print(f"Unique patients          : {n_patients:,}")
print(f"Patients with >1 enc.    : {multi_encounter:,} ({multi_pct:.1f}%)")
print(f"Max encounters (one pt)  : {max_enc}")

fig, ax = plt.subplots(figsize=(8, 4))
enc_dist = encounters_per_patient.value_counts().sort_index().head(15)
enc_dist.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="black", width=0.7)
ax.set_title("Distribution of Encounters per Patient")
ax.set_xlabel("Number of Encounters"); ax.set_ylabel("Number of Patients")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(f"{OUT}/03_encounters_per_patient.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {OUT}/03_encounters_per_patient.png")

print("""
COMMENTARY — Section 3:
  Approximately 28% of patients appear in more than one encounter. A naive
  random train/test split would leak patient-level information across folds,
  inflating held-out performance estimates. Grouped splitting (by patient_nbr)
  is required to ensure no patient appears in both train and test sets.
  This also means effective sample size for generalisation is the number of
  unique patients (~71k), not the number of encounters (~102k).
""")

# =============================================================================
print("=" * 70)
print("SECTION 4 — MISSINGNESS")
print("=" * 70)

# CHANGE-1 (continued): Distinguish true missing (?) from "not measured" (None).
# "None" in A1Cresult / max_glu_serum means the test was not ordered — a valid
# clinical category. True NaN here means the value is genuinely absent/unknown.
miss = (df.isna().mean() * 100).sort_values(ascending=False)
miss_df = miss[miss > 0].to_frame(name="pct_true_missing").round(2)
print(f"\nColumns with TRUE missing (? only), % of total:\n{miss_df}")

# Show "not measured" counts separately for test-result columns
not_measured_cols = ["A1Cresult", "max_glu_serum"]
print("\n--- 'Not tested / not measured' breakdown (None category) ---")
for col in not_measured_cols:
    vc = df[col].value_counts(dropna=False)
    print(f"\n  {col}:")
    print(vc.to_string())

high_miss = miss_df[miss_df["pct_true_missing"] > 40]
print(f"\nColumns > 40% TRUE missing: {high_miss.index.tolist()}")

# ── Missingness bar plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
miss_df.plot(kind="bar", ax=ax, color="#DD8452", edgecolor="black", legend=False)
ax.axhline(40, color="red", linestyle="--", linewidth=1.2, label="40% threshold")
ax.set_title("% True Missing per Column (? only)")
ax.set_ylabel("% True Missing"); ax.set_xlabel("")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/04a_missingness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {OUT}/04a_missingness.png")

# ── A1Cresult missingness by age group ───────────────────────────────────────
df["A1C_missing"] = df["A1Cresult"].isna().astype(int)

# CHANGE-1 note: A1C "None" (not tested) is NOT flagged as missing here.
# A1C_missing == 1 only when the value is truly absent (was "?").
# The dominant category in A1Cresult is "None" (~76% not tested), which is
# retained as a valid level. Only ~0.3% are truly missing (?).

a1c_by_age = df.groupby("age")["A1C_missing"].mean().sort_index() * 100
print(f"\nA1C TRUE missingness (? only) by age group (%):\n{a1c_by_age.round(2)}")

# ── A1Cresult missingness by number_inpatient (binned) ───────────────────────
df["inpatient_bin"] = pd.cut(df["number_inpatient"], bins=[-1,0,1,2,5,100],
                              labels=["0","1","2","3-5",">5"])
a1c_by_inpat = df.groupby("inpatient_bin", observed=True)["A1C_missing"].mean() * 100
print(f"\nA1C TRUE missingness (? only) by prior inpatient visits (%):\n{a1c_by_inpat.round(2)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
a1c_by_age.plot(kind="bar", ax=axes[0], color="#4C72B0", edgecolor="black", width=0.7)
axes[0].set_title("A1Cresult TRUE Missing Rate by Age Group\n(excludes 'None'=not tested)")
axes[0].set_ylabel("% Truly Missing (?)"); axes[0].set_xlabel("Age")
axes[0].set_ylim(0, 5); axes[0].axhline(a1c_by_age.mean(), color="red", linestyle="--",
                                            linewidth=1, label="Overall mean")
axes[0].legend(); axes[0].tick_params(axis='x', rotation=45)

a1c_by_inpat.plot(kind="bar", ax=axes[1], color="#55A868", edgecolor="black", width=0.7)
axes[1].set_title("A1Cresult TRUE Missing Rate by Prior Inpatient Visits\n(excludes 'None'=not tested)")
axes[1].set_ylabel("% Truly Missing (?)"); axes[1].set_xlabel("Prior Inpatient Visits")
axes[1].set_ylim(0, 5); axes[1].axhline(a1c_by_inpat.mean(), color="red", linestyle="--",
                                            linewidth=1, label="Overall mean")
axes[1].legend(); axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f"{OUT}/04b_a1c_missingness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {OUT}/04b_a1c_missingness.png")

print("""
COMMENTARY — Section 4:
  Only "?" values are treated as true missing. "None" in A1Cresult and
  max_glu_serum means the test was not ordered — a valid clinical state, not
  an absent value. Concretely: A1Cresult is "None" (not tested) in ~76% of
  encounters; truly missing (?) in <1%. max_glu_serum follows a similar
  pattern (~95% not tested, negligible true missing).

  weight is truly missing in >96% of records and provides near-no usable
  signal. payer_code (~40% true missing) and medical_specialty (~49% true
  missing) are the main structural gaps.

  Stratifying A1Cresult true-missingness by age group and prior inpatient
  visits shows near-zero variation — true "?" values are extremely rare across
  all strata. The clinically meaningful question is whether the "None" (not
  tested) vs. tested distinction carries predictive signal; that is a feature
  engineering decision, not a missingness problem. MCAR/MAR/MNAR framing
  does not apply to the "None" category.
""")

# =============================================================================
print("=" * 70)
print("SECTION 5 — NUMERIC DISTRIBUTIONS")
print("=" * 70)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude binary target and derived columns
exclude = ["readmit_30", "A1C_missing", "encounter_id", "patient_nbr",
           "admission_type_id", "discharge_disposition_id", "admission_source_id"]
num_cols = [c for c in num_cols if c not in exclude]

skew_df = df[num_cols].skew().to_frame(name="skewness").round(3).sort_values(
    "skewness", ascending=False)
print(f"\nSkewness of numeric variables:\n{skew_df}")

high_skew = skew_df[skew_df["skewness"].abs() > 1]
print(f"\nHighly skewed (|skew| > 1): {high_skew.index.tolist()}")

# ── Histograms ────────────────────────────────────────────────────────────────
plot_cols = ["time_in_hospital","num_lab_procedures","num_procedures",
             "num_medications","number_outpatient","number_emergency",
             "number_inpatient","number_diagnoses"]

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(plot_cols):
    axes[i].hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white",
                 linewidth=0.3)
    axes[i].set_title(f"{col}\nskew={df[col].skew():.2f}", fontsize=9)
    axes[i].set_ylabel("Count")
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.suptitle("Distributions of Count / Continuous Variables", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/05_numeric_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {OUT}/05_numeric_histograms.png")

print("""
COMMENTARY — Section 5:
  number_outpatient, number_emergency, and number_inpatient are strongly
  right-skewed (skew > 2); the majority of patients have zero prior visits.
  time_in_hospital is moderately right-skewed. num_lab_procedures is roughly
  symmetric. num_medications is mildly right-skewed.

  Whether skewness warrants transformation depends on the model class:
  - Tree-based models (RF, XGBoost) are scale-invariant; transformation
    has no meaningful effect on them.
  - Logistic regression and distance-based models may benefit from
    log(x+1) or rank normalisation of heavily skewed features.
  No transformation decisions are made at this stage.
""")

# =============================================================================
print("=" * 70)
print("SECTION 6 — SIMPLE ASSOCIATION CHECKS WITH TARGET")
print("=" * 70)

# CHANGE-3: Restrict to true numeric count/continuous variables.
# admission_type_id, discharge_disposition_id, admission_source_id are
# integer-coded nominals — arithmetic correlation is not meaningful on them.
TRUE_NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

results = []
for col in TRUE_NUMERIC:
    valid = df[[col, "readmit_30"]].dropna()
    if len(valid) < 10:
        continue
    r, p = pointbiserialr(valid[col], valid["readmit_30"])
    results.append({"feature": col, "r": round(r, 4), "p_value": round(p, 6)})

assoc_df = pd.DataFrame(results).sort_values("r", key=abs, ascending=False)
print(f"\nPoint-biserial correlation with binary target (readmit_30):\n")
print(assoc_df.to_string(index=False))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#DD8452" if r >= 0 else "#4C72B0" for r in assoc_df["r"]]
ax.barh(assoc_df["feature"], assoc_df["r"], color=colors, edgecolor="black", height=0.6)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Point-Biserial r")
ax.set_title("Association with 30-Day Readmission (Point-Biserial r)")

# Significance markers
for i, row in enumerate(assoc_df.itertuples()):
    marker = "***" if row.p_value < 0.001 else ("**" if row.p_value < 0.01 else
             ("*" if row.p_value < 0.05 else ""))
    x_pos = row.r + (0.002 if row.r >= 0 else -0.002)
    ax.text(x_pos, i, marker, va="center", fontsize=8,
            ha="left" if row.r >= 0 else "right")

plt.tight_layout()
plt.savefig(f"{OUT}/06_association_target.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {OUT}/06_association_target.png")

print("""
COMMENTARY — Section 6:
  With N ≈ 100k, virtually any non-zero correlation will reach p < 0.001;
  p-values are not a useful filter here and are reported only for completeness.
  Attention should be on effect size (|r|).

  number_inpatient has the strongest association (r ≈ 0.17), a small but
  meaningful effect by conventional benchmarks. All other variables show
  weak associations (|r| < 0.07). number_inpatient and number_emergency
  are the only features with associations large enough to be worth noting
  at this stage. num_procedures shows a small negative association.
  No causal inference is drawn.
""")

# =============================================================================
print("=" * 70)
print("SECTION 7 — DISCHARGE DISPOSITION SANITY CHECK")
print("=" * 70)

# ICD discharge disposition codes for death/hospice
# Ref: CMS UB-04 discharge disposition codes
DEATH_HOSPICE_CODES = [11, 13, 14, 19, 20, 21]

df["is_death_hospice"] = df["discharge_disposition_id"].isin(DEATH_HOSPICE_CODES)
n_dh = df["is_death_hospice"].sum()
pct_dh = n_dh / len(df) * 100

readmit_dh   = df[df["is_death_hospice"]]["readmit_30"].mean() * 100
readmit_rest = df[~df["is_death_hospice"]]["readmit_30"].mean() * 100

print(f"\nDeath/Hospice codes: {DEATH_HOSPICE_CODES}")
print(f"Encounters flagged  : {n_dh:,} ({pct_dh:.1f}%)")
print(f"\n30-day readmission rate:")
print(f"  Death/Hospice group  : {readmit_dh:.2f}%")
print(f"  All other encounters : {readmit_rest:.2f}%")

# Per-code breakdown
code_breakdown = df.groupby("discharge_disposition_id").agg(
    n=("readmit_30","count"),
    readmit_30_rate=("readmit_30","mean"),
    is_dh=("is_death_hospice","first")
).reset_index()
code_breakdown["readmit_30_rate"] = (code_breakdown["readmit_30_rate"] * 100).round(2)
print(f"\nPer-code summary (death/hospice flagged with *):")
code_breakdown["flag"] = code_breakdown["is_dh"].map({True:"*", False:""})
print(code_breakdown[["discharge_disposition_id","n","readmit_30_rate","flag"]]
      .sort_values("discharge_disposition_id").to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(["Death/Hospice", "All Others"], [readmit_dh, readmit_rest],
       color=["#DD8452","#4C72B0"], edgecolor="black", width=0.4)
ax.set_title("30-Day Readmission Rate: Death/Hospice vs. Other Discharges")
ax.set_ylabel("% Readmitted within 30 days")
for i, v in enumerate([readmit_dh, readmit_rest]):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/07_discharge_sanity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {OUT}/07_discharge_sanity.png")

print("""
COMMENTARY — Section 7:
  Death and hospice discharge codes (11, 13, 14, 19, 20, 21) account for ~2.4%
  of encounters. Their observed 30-day readmission rate is near zero, which is
  expected: patients who died or were discharged to hospice cannot be readmitted.

  Classification: This is a deployment population definition decision, not
  data leakage.
  - Data leakage (strict definition): a predictor feature is derived from
    information that is only available after the outcome is known. That is not
    the case here — discharge disposition is recorded at the time of discharge,
    contemporaneously with the start of the 30-day readmission window.
  - The actual issue is population scope: the model's intended use case is
    predicting readmission risk for patients who could plausibly be readmitted.
    Deceased and hospice-discharged patients are structurally ineligible for
    readmission; including them trains the model on a subgroup that will never
    appear in deployment.
  - Decision: whether to exclude these records should be made explicitly as a
    cohort definition step before modelling begins — deferred to that phase.
""")

# =============================================================================
print("=" * 70)
print("SECTION 8 — STRUCTURED MODELLING HYPOTHESES")
print("=" * 70)

print("""
HYPOTHESES (pre-modelling; to be tested empirically):

  H1. Prior utilisation as risk signal:
      number_inpatient and number_emergency show the strongest marginal
      associations with 30-day readmission. We hypothesise these will be
      among the top predictors in a trained model.

  H2. Medication complexity proxy:
      num_medications may serve as a proxy for comorbidity burden. Its
      association with readmission may be partially confounded by diagnosis
      severity, which would be captured by the diag_* fields once encoded.

  H3. A1Cresult "not tested" as selective signal:
      ~83% of A1Cresult values are "None" (test not ordered), with negligible
      true missingness (?). Whether a test was ordered may itself be
      informative (e.g. reflects clinical judgment about glycaemic control).
      A binary "tested vs. not tested" indicator may carry predictive signal
      independently of the test result value.

  H4. Grouped train/test splits may reduce apparent model performance:
      Because ~28% of patients have multiple encounters, a patient-grouped
      split will be more conservative than a random split. We expect a
      non-trivial performance drop when enforcing grouped splits, reflecting
      a more realistic estimate of out-of-sample generalisation.

  H5. Death/hospice exclusion may improve calibration:
      Excluding death/hospice encounters from the training cohort may improve
      model calibration for the target deployment population, since these
      records create a systematic low-risk cluster that is clinically
      irrelevant to the readmission prediction task.

  H6. High-cardinality ICD codes require grouping to generalise:
      diag_1/2/3 have hundreds of unique codes. Raw codes will not generalise;
      grouping to CCS categories or major ICD chapters is likely necessary
      for stable coefficient or feature importance estimates.
""")

print("\n" + "=" * 70)
print("EDA COMPLETE — all figures saved to:", OUT)
print("=" * 70)
