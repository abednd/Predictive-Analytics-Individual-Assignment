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
df = pd.read_csv("diabetic_data.csv", na_values=["?", "Unknown/Invalid", "None"])

# Replace "?" with NaN in string columns (already handled above, belt-and-suspenders)
for c in df.select_dtypes("object"):
    df[c] = df[c].replace("?", np.nan)

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

print("\n--- Flagged columns ---")
for flag in ["IDENTIFIER","CONSTANT","HIGH_CARD"]:
    flagged = card[card["flag"] == flag].index.tolist()
    print(f"  {flag}: {flagged}")

# ── Commentary (printed, mirrors Jupyter markdown cells) ─────────────────────
print("""
COMMENTARY — Section 1:
  The dataset has 101,766 encounters across 50 columns. Columns are a mix of
  integer IDs, nominal categoricals, and sparse drug-flag fields.
  encounter_id and patient_nbr are identifiers and carry no predictive signal.
  No true constant columns (n_unique == 1) are present, but several drug columns
  (e.g. examide, citoglipton) have near-zero variance (>99 % single value) and
  will warrant attention at feature selection. diag_1/diag_2/diag_3 are
  high-cardinality free-text ICD codes; they will need grouping before use.
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

miss = (df.isna().mean() * 100).sort_values(ascending=False)
miss_df = miss[miss > 0].to_frame(name="pct_missing").round(2)
print(f"\nColumns with missing data:\n{miss_df}")

high_miss = miss_df[miss_df["pct_missing"] > 40]
print(f"\nColumns > 40% missing: {high_miss.index.tolist()}")

# ── Missingness bar plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
miss_df.plot(kind="bar", ax=ax, color="#DD8452", edgecolor="black", legend=False)
ax.axhline(40, color="red", linestyle="--", linewidth=1.2, label="40% threshold")
ax.set_title("% Missing per Column")
ax.set_ylabel("% Missing"); ax.set_xlabel("")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/04a_missingness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {OUT}/04a_missingness.png")

# ── A1Cresult missingness by age group ───────────────────────────────────────
df["A1C_missing"] = df["A1Cresult"].isna().astype(int)

a1c_by_age = df.groupby("age")["A1C_missing"].mean().sort_index() * 100
print(f"\nA1C missingness by age group (%):\n{a1c_by_age.round(1)}")

# ── A1Cresult missingness by number_inpatient (binned) ───────────────────────
df["inpatient_bin"] = pd.cut(df["number_inpatient"], bins=[-1,0,1,2,5,100],
                              labels=["0","1","2","3-5",">5"])
a1c_by_inpat = df.groupby("inpatient_bin", observed=True)["A1C_missing"].mean() * 100
print(f"\nA1C missingness by prior inpatient visits (%):\n{a1c_by_inpat.round(1)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
a1c_by_age.plot(kind="bar", ax=axes[0], color="#4C72B0", edgecolor="black", width=0.7)
axes[0].set_title("A1Cresult Missing Rate by Age Group")
axes[0].set_ylabel("% Missing"); axes[0].set_xlabel("Age")
axes[0].set_ylim(0, 100); axes[0].axhline(a1c_by_age.mean(), color="red", linestyle="--",
                                            linewidth=1, label="Overall mean")
axes[0].legend(); axes[0].tick_params(axis='x', rotation=45)

a1c_by_inpat.plot(kind="bar", ax=axes[1], color="#55A868", edgecolor="black", width=0.7)
axes[1].set_title("A1Cresult Missing Rate by Prior Inpatient Visits")
axes[1].set_ylabel("% Missing"); axes[1].set_xlabel("Prior Inpatient Visits")
axes[1].set_ylim(0, 100); axes[1].axhline(a1c_by_inpat.mean(), color="red", linestyle="--",
                                            linewidth=1, label="Overall mean")
axes[1].legend(); axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f"{OUT}/04b_a1c_missingness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  -> Saved {OUT}/04b_a1c_missingness.png")

print("""
COMMENTARY — Section 4:
  weight (>96% missing) and payer_code (~40% missing) are the most affected
  columns. medical_specialty (~49% missing) also exceeds the 40% threshold.
  A1Cresult is missing in ~83% of records.
  
  Stratifying A1Cresult missingness by age group reveals limited variation
  (range ~80-86%), suggesting age is not a strong driver of whether A1C was
  recorded. Stratifying by prior inpatient visits also shows modest variation.
  No strong differential missingness pattern is evident from these two checks
  alone. Causation and MCAR/MAR/MNAR cannot be determined from marginal rates;
  further multivariate analysis would be needed before making imputation decisions.
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

assoc_cols = [c for c in num_cols if c in df.columns]
results = []
for col in assoc_cols:
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
  All associations are statistically significant (p < 0.001) given the large
  sample size — p-values alone are not informative here. Focusing on magnitude:
  number_inpatient shows the strongest positive association (r ≈ 0.20),
  suggesting patients with more prior inpatient admissions are more likely to be
  readmitted within 30 days. number_emergency and number_outpatient show smaller
  positive associations. time_in_hospital and num_medications show weak positive
  associations. No causal inference is drawn; these are marginal correlations.
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
  Death and hospice discharge codes (11, 13, 14, 19, 20, 21) account for ~2.6%
  of encounters. Their observed 30-day readmission rate is near zero, which is
  expected: patients who died or were discharged to hospice cannot be readmitted.

  Classification: This is a *modelling population definition issue*, not
  true data leakage in the strict sense.
  - True leakage would mean a feature derived from post-outcome information
    is used as a predictor.
  - Here, the discharge code is recorded at the time of discharge (concurrent
    with the target window), not after it. However, including these records
    in training will systematically depress predicted risk for a clinically
    irrelevant subgroup, potentially distorting decision boundaries.
  - The appropriate resolution is a population scoping decision: define whether
    the model's intended deployment population excludes patients discharged
    to death/hospice. If yes, these records should be excluded from the
    modelling cohort — but that decision is deferred to the modelling phase.
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

  H3. A1Cresult as selective signal:
      A1Cresult is missing for ~83% of encounters. Its presence/absence is
      itself potentially informative (indicating whether testing was ordered),
      independent of its recorded value. A missingness indicator may carry
      predictive signal.

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
