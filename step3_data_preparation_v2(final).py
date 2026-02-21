# =============================================================================
# Step 3 — Data Preparation Pipeline
# =============================================================================
# Run in Jupyter. Place diabetic_data.csv in the same directory as this script.
#
# Annotations throughout:
#   [EDA]        decision grounded directly in EDA findings
#   [ASSUMPTION] working assumption to validate at modelling stage
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, OneHotEncoder
)
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42

# =============================================================================
# SECTION 0 — LOAD
# =============================================================================
# [EDA] Only "?" is true missing. "None" is a valid category meaning
#       "test not ordered" (e.g. A1Cresult, max_glu_serum).
df = pd.read_csv(
    "diabetic_data.csv",
    na_values=["?"],
    keep_default_na=False,
    low_memory=False
)

print("=" * 70)
print("STEP 3 — DATA PREPARATION PIPELINE")
print("=" * 70)
print(f"\n[0] Raw dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")


# =============================================================================
# SECTION 1 — TARGET VARIABLE
# =============================================================================
# [EDA] Binary target: 1 = readmitted within <30 days, 0 = all other outcomes.
df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
pos_pct = df["readmit_30"].mean() * 100
print(f"\n[1] Target created: readmit_30")
print(f"    Positive class: {df['readmit_30'].sum():,} ({pos_pct:.1f}%)")
print(f"    Negative class: {(df['readmit_30'] == 0).sum():,} ({100-pos_pct:.1f}%)")


# =============================================================================
# SECTION 2 — FEATURE EXCLUSIONS
# =============================================================================
# [EDA] Columns removed before any modelling:
#   encounter_id            : row identifier, no predictive content
#   patient_nbr             : used only for grouped splitting; never a feature
#   examide, citoglipton    : confirmed constant (all "No" across 101,766 rows)
#   readmitted              : source column for target; drop to prevent leakage

DROP_COLS = ["encounter_id", "examide", "citoglipton", "readmitted"]

# Keep patient_nbr separate for split grouping, then exclude from features
patient_ids = df["patient_nbr"].copy()

df_features = df.drop(columns=DROP_COLS + ["patient_nbr", "readmit_30"])
y = df["readmit_30"].copy()

print(f"\n[2] Dropped columns: {DROP_COLS + ['patient_nbr']}")
print(f"    Feature matrix shape: {df_features.shape}")


# =============================================================================
# SECTION 3 — FEATURE-TYPE INVENTORY
# =============================================================================

# [EDA] True continuous / count variables; no missing values in this dataset.
NUMERIC_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# [EDA] "age" is an ordinal string band — treated as ordinal (not OHE).
AGE_COL = ["age"]
AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]

# [EDA] max_glu_serum and A1Cresult: "None" = not tested (valid category).
#       Kept as categorical; tested-vs-not indicators added in Section 4.
# [EDA] "gender" contains "Unknown/Invalid" — kept as its own category.
# [ASSUMPTION] admission_type_id, discharge_disposition_id, admission_source_id
#              are integer-coded nominals with no verified ordinal structure;
#              treated as categorical via OHE. Revisit if domain mapping available.
CATEGORICAL_COLS = [
    "race",
    "gender",
    "weight",
    "payer_code",
    "medical_specialty",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
]

# [EDA] High-cardinality ICD fields; grouped into chapters in Section 5.
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

# [EDA] Medication columns: raw levels are No / Steady / Up / Down.
# [ASSUMPTION] Treated as ordinal with 3 levels: No < Steady < Changed.
#              Up and Down are both explicitly remapped to "Changed" in
#              Section 4b before encoding, so code and comments are consistent.
#              Revisit if directional signal (Up vs Down) matters.
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

print(f"\n[3] Feature-type groupings:")
print(f"    Numeric        ({len(NUMERIC_COLS):2d} cols): {NUMERIC_COLS}")
print(f"    Age (ordinal)  ({ len(AGE_COL):2d} col ): {AGE_COL}")
print(f"    Categorical    ({len(CATEGORICAL_COLS):2d} cols): {CATEGORICAL_COLS}")
print(f"    Diagnosis      ({ len(DIAG_COLS):2d} cols): {DIAG_COLS}")
print(f"    Medication     ({len(MEDICATION_COLS):2d} cols): {MEDICATION_COLS}")

# Sanity check: every feature column is assigned exactly once
all_assigned = set(
    NUMERIC_COLS + AGE_COL + CATEGORICAL_COLS + DIAG_COLS + MEDICATION_COLS
)
all_features = set(df_features.columns)
unassigned = all_features - all_assigned
print(f"\n    Column assignment check: "
      f"{'All columns assigned ✓' if not unassigned else f'UNASSIGNED: {unassigned}'}")


# =============================================================================
# SECTION 4 — PRE-SPLIT FEATURE ENGINEERING
# =============================================================================
# Safe to compute on the full dataset: derived only from row values, not target.

# [EDA] A1Cresult ~83% "None" (not ordered); max_glu_serum ~95% "None".
#       Original columns are retained as categorical features.
#       Binary indicators capture whether the test was ordered at all.
# [ASSUMPTION] Tested-vs-not may carry independent predictive signal
#              (reflects clinical decision at time of encounter). To be evaluated.

df_features["A1C_tested"] = (df_features["A1Cresult"]    != "None").astype(int)
df_features["glu_tested"] = (df_features["max_glu_serum"] != "None").astype(int)

NUMERIC_COLS_EXTENDED = NUMERIC_COLS + ["A1C_tested", "glu_tested"]

print(f"\n[4] Lab-test indicators added (tested=1 / not tested=0):")
print(f"    A1C_tested : tested={df_features['A1C_tested'].sum():,}  "
      f"not tested={(df_features['A1C_tested']==0).sum():,}")
print(f"    glu_tested : tested={df_features['glu_tested'].sum():,}  "
      f"not tested={(df_features['glu_tested']==0).sum():,}")

# ── 4b. Medication Up/Down → Changed (explicit remap before encoding) ─────────
# [ASSUMPTION] Up and Down both mean "dose was modified"; direction is collapsed
#              here so the ordinal categories in the pipeline are exactly
#              No < Steady < Changed. If directional signal is needed later,
#              revert this map and use 4-level encoding instead.
MED_REMAP = {"Up": "Changed", "Down": "Changed"}
for col in MEDICATION_COLS:
    df_features[col] = df_features[col].replace(MED_REMAP)

# Spot-check: confirm only {No, Steady, Changed} remain
med_remaining = set()
for col in MEDICATION_COLS:
    med_remaining |= set(df_features[col].dropna().unique())
print(f"\n[4b] Medication levels after remap: {sorted(med_remaining)}"
      f"  (expected: ['Changed', 'No', 'Steady'])")


# =============================================================================
# SECTION 5 — DIAGNOSIS CODE GROUPING
# =============================================================================
# [EDA] diag_1/2/3 each have 700–790 unique raw ICD-9 codes.
# [ASSUMPTION] ICD-9 chapter-level grouping used as the minimal reproducible
#              baseline (18 chapters + V/E supplementary codes).
#              Finer groupings (CCS, diabetes-specific flags) can be swapped in
#              later without changing pipeline structure downstream.

def map_icd9_chapter(code):
    """Map a raw ICD-9 code string to a broad chapter label."""
    if pd.isna(code) or str(code).strip() in ("", "None"):
        return "Missing"
    code = str(code).strip()
    if code.startswith("V"):
        return "V_supplementary"
    if code.startswith("E"):
        return "E_external"
    try:
        num = float(code.split(".")[0])
    except ValueError:
        return "Other"
    if   1   <= num <= 139: return "Infectious"
    elif 140 <= num <= 239: return "Neoplasms"
    elif 240 <= num <= 279: return "Endocrine_metabolic"
    elif 280 <= num <= 289: return "Blood"
    elif 290 <= num <= 319: return "Mental"
    elif 320 <= num <= 389: return "Nervous_sensory"
    elif 390 <= num <= 459: return "Circulatory"
    elif 460 <= num <= 519: return "Respiratory"
    elif 520 <= num <= 579: return "Digestive"
    elif 580 <= num <= 629: return "Genitourinary"
    elif 630 <= num <= 679: return "Pregnancy"
    elif 680 <= num <= 709: return "Skin"
    elif 710 <= num <= 739: return "Musculoskeletal"
    elif 740 <= num <= 759: return "Congenital"
    elif 760 <= num <= 779: return "Perinatal"
    elif 780 <= num <= 799: return "Symptoms_signs"
    elif 800 <= num <= 999: return "Injury_poisoning"
    else:                    return "Other"

for col in DIAG_COLS:
    df_features[col + "_chapter"] = df_features[col].apply(map_icd9_chapter)
    df_features.drop(columns=[col], inplace=True)

DIAG_CHAPTER_COLS = [c + "_chapter" for c in DIAG_COLS]

print(f"\n[5] ICD-9 chapter grouping applied:")
for col in DIAG_CHAPTER_COLS:
    n_ch = df_features[col].nunique()
    top3 = df_features[col].value_counts().head(3).to_dict()
    print(f"    {col}: {n_ch} chapters  |  top 3: {top3}")


# =============================================================================
# SECTION 6 — LEAKAGE-SAFE GROUPED SPLITS
# =============================================================================
# [EDA] 23.5% of patients have multiple encounters. GroupShuffleSplit ensures
#       each patient appears in exactly one split, preventing patient-level
#       information from leaking across train / val / test.

# Step 6a: train (70%) vs. temp (30%)
gss_main = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
train_idx, temp_idx = next(
    gss_main.split(df_features, y, groups=patient_ids)
)

X_train_raw  = df_features.iloc[train_idx].reset_index(drop=True)
y_train      = y.iloc[train_idx].reset_index(drop=True)
groups_train = patient_ids.iloc[train_idx].reset_index(drop=True)

X_temp       = df_features.iloc[temp_idx].reset_index(drop=True)
y_temp       = y.iloc[temp_idx].reset_index(drop=True)
groups_temp  = patient_ids.iloc[temp_idx].reset_index(drop=True)

# Step 6b: val (50% of temp ≈ 15% overall) vs. test (50% of temp ≈ 15% overall)
gss_valtest = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
val_idx, test_idx = next(
    gss_valtest.split(X_temp, y_temp, groups=groups_temp)
)

X_val_raw  = X_temp.iloc[val_idx].reset_index(drop=True)
y_val      = y_temp.iloc[val_idx].reset_index(drop=True)
groups_val = groups_temp.iloc[val_idx].reset_index(drop=True)

X_test_raw  = X_temp.iloc[test_idx].reset_index(drop=True)
y_test      = y_temp.iloc[test_idx].reset_index(drop=True)
groups_test = groups_temp.iloc[test_idx].reset_index(drop=True)

print(f"\n[6] Grouped splits created:")
print(f"    {'Split':<12} {'Rows':>7}  {'Patients':>9}  {'% Positive':>11}")
print(f"    {'-'*44}")
for name, X_r, yy, g in [
    ("Train",      X_train_raw, y_train, groups_train),
    ("Validation", X_val_raw,   y_val,   groups_val),
    ("Test",       X_test_raw,  y_test,  groups_test),
]:
    print(f"    {name:<12} {len(X_r):>7,}  {g.nunique():>9,}  {yy.mean()*100:>10.1f}%")

# ── Zero-overlap verification ─────────────────────────────────────────────────
train_pts = set(groups_train)
val_pts   = set(groups_val)
test_pts  = set(groups_test)

overlap_tv = train_pts & val_pts
overlap_tt = train_pts & test_pts
overlap_vt = val_pts   & test_pts

print(f"\n    Patient overlap checks:")
print(f"    Train ∩ Val  : {len(overlap_tv):>4} patients — "
      f"{'✓ No overlap' if not overlap_tv else '✗ OVERLAP DETECTED'}")
print(f"    Train ∩ Test : {len(overlap_tt):>4} patients — "
      f"{'✓ No overlap' if not overlap_tt else '✗ OVERLAP DETECTED'}")
print(f"    Val ∩ Test   : {len(overlap_vt):>4} patients — "
      f"{'✓ No overlap' if not overlap_vt else '✗ OVERLAP DETECTED'}")


# =============================================================================
# SECTION 7 — PREPROCESSING PIPELINE (ColumnTransformer)
# =============================================================================
# CRITICAL: The preprocessor is fit ONLY on X_train_raw.
#           X_val_raw and X_test_raw are transformed using the fitted object.
#           No statistics from val/test leak into the pipeline.

# ── Numeric ───────────────────────────────────────────────────────────────────
# [EDA] No true missing in numeric cols; median imputer is defensive fallback.
# [ASSUMPTION] StandardScaler applied for model-agnostic compatibility.
#              Irrelevant for tree models; harmless to include.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

# ── Age ordinal ───────────────────────────────────────────────────────────────
# Decade bands have a clear natural order; encode as 0–9.
age_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[AGE_ORDER],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

# ── Categorical (nominal) ─────────────────────────────────────────────────────
# [EDA] True missing (?) → "Missing" fill before OHE.
#       handle_unknown="ignore" → all-zero vector for unseen categories at inference.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# ── Medication ordinal ────────────────────────────────────────────────────────
# [ASSUMPTION] 3-level ordinal: No < Steady < Changed.
#              Up/Down were remapped to "Changed" in Section 4b, so the
#              encoder categories match the data exactly — no contradiction.
MED_CATEGORIES = [["No", "Steady", "Changed"]] * len(MEDICATION_COLS)

medication_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="No")),
    ("encoder", OrdinalEncoder(
        categories=MED_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

# ── Diagnosis chapters ────────────────────────────────────────────────────────
# Chapter labels are now nominal strings; OHE them.
diag_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# ── Assemble ColumnTransformer ────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric",     numeric_pipeline,     NUMERIC_COLS_EXTENDED),
        ("age",         age_pipeline,          AGE_COL),
        ("categorical", categorical_pipeline,  CATEGORICAL_COLS),
        ("medication",  medication_pipeline,   MEDICATION_COLS),
        ("diagnosis",   diag_pipeline,         DIAG_CHAPTER_COLS),
    ],
    remainder="drop",                     # anything not listed is excluded
    verbose_feature_names_out=True,
)

# ── Fit on train; transform all splits ───────────────────────────────────────
print(f"\n[7] Fitting preprocessor on training set only …")
preprocessor.fit(X_train_raw)
print(f"    Fit complete.")

X_train = pd.DataFrame(
    preprocessor.transform(X_train_raw),
    columns=preprocessor.get_feature_names_out()
)
X_val = pd.DataFrame(
    preprocessor.transform(X_val_raw),
    columns=preprocessor.get_feature_names_out()
)
X_test = pd.DataFrame(
    preprocessor.transform(X_test_raw),
    columns=preprocessor.get_feature_names_out()
)


# =============================================================================
# SECTION 8 — OUTPUT CHECKS
# =============================================================================

print(f"\n{'=' * 70}")
print("FINAL OUTPUT CHECKS")
print(f"{'=' * 70}")

print(f"\nProcessed shapes:")
print(f"    X_train : {X_train.shape}  |  y_train : {y_train.shape}")
print(f"    X_val   : {X_val.shape}  |  y_val   : {y_val.shape}")
print(f"    X_test  : {X_test.shape}  |  y_test  : {y_test.shape}")

print(f"\nNaN check (all must be 0):")
for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
    n_nan = X.isna().sum().sum()
    print(f"    {name} NaNs : {n_nan}  {'✓' if n_nan == 0 else '✗ NaNs PRESENT'}")

print(f"\nFeature breakdown:")
n_num  = len(NUMERIC_COLS_EXTENDED)
n_age  = 1
n_cat  = X_train.filter(like="categorical__").shape[1]
n_med  = len(MEDICATION_COLS)
n_diag = X_train.filter(like="diagnosis__").shape[1]
print(f"    Numeric (incl. lab indicators)  : {n_num}")
print(f"    Age (ordinal → 1 col)           : {n_age}")
print(f"    Categorical (OHE expanded)      : {n_cat}")
print(f"    Medication (ordinal → 1 col ea) : {n_med}")
print(f"    Diagnosis chapters (OHE)        : {n_diag}")
print(f"    ─────────────────────────────────")
print(f"    Total features                  : {X_train.shape[1]}")

print(f"\nTarget rate by split:")
for name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"    {name:<10}: {yy.mean()*100:.2f}% positive")

print(f"\n{'=' * 70}")
print("PIPELINE READY")
print(f"    Objects available for modelling:")
print(f"      Features : X_train, X_val, X_test")
print(f"      Targets  : y_train, y_val, y_test")
print(f"      Pipeline : preprocessor  (fitted on train only)")
print(f"{'=' * 70}")


# =============================================================================
# SECTION 9 — PERSIST PIPELINE AND SPLIT METADATA
# =============================================================================
import joblib

# ── 9a. Save fitted preprocessor ──────────────────────────────────────────────
joblib.dump(preprocessor, "preprocessor.joblib")
print(f"\n[9a] Fitted preprocessor saved → preprocessor.joblib")

# ── 9b. Save split patient IDs and row indices ────────────────────────────────
# Storing patient IDs lets us reconstruct or audit splits without re-running.
split_meta = pd.DataFrame({
    "original_index": list(train_idx) + list(temp_idx[val_idx]) + list(temp_idx[test_idx]),
    "patient_nbr":    list(patient_ids.iloc[train_idx])
                      + list(patient_ids.iloc[temp_idx].iloc[val_idx])
                      + list(patient_ids.iloc[temp_idx].iloc[test_idx]),
    "split":          (["train"] * len(train_idx)
                       + ["val"]  * len(val_idx)
                       + ["test"] * len(test_idx)),
})
split_meta.to_csv("split_patient_ids.csv", index=False)
print(f"[9b] Split patient IDs + indices saved → split_patient_ids.csv")
print(f"     Rows: train={len(train_idx):,}  val={len(val_idx):,}  "
      f"test={len(test_idx):,}  total={len(split_meta):,}")


# =============================================================================
# SECTION 9 — DECISION LOG
# =============================================================================
print("""
DECISION LOG
─────────────────────────────────────────────────────────────────────────────
GROUNDED IN EDA FINDINGS:
  [EDA-1]  "?" only = true missing; "None" retained as valid category
  [EDA-2]  encounter_id / patient_nbr / examide / citoglipton excluded
  [EDA-3]  GroupShuffleSplit by patient_nbr (23.5% multi-encounter patients)
  [EDA-4]  A1C_tested + glu_tested indicators (83% / 95% "None" in source cols)
  [EDA-5]  Integer-coded ID columns treated as nominal, not numeric
  [EDA-6]  ICD columns chapter-grouped (EDA: 700–790 unique codes per field)
  [EDA-7]  Numeric columns confirmed no true missing; imputer is fallback only

WORKING ASSUMPTIONS — VALIDATE AT MODELLING STAGE:
  [A-1]  Medication ordinal: No < Steady < Changed; Up/Down explicitly remapped
         to "Changed" before encoding — code and comments now consistent
  [A-2]  StandardScaler applied uniformly; harmless for trees, useful for linear
  [A-3]  Nominal IDs treated via OHE; ordinal structure possible with domain map
  [A-4]  ICD chapter grouping is baseline; CCS / diabetes-specific flags may help
  [A-5]  A1C_tested / glu_tested assumed to carry signal independent of result
─────────────────────────────────────────────────────────────────────────────
""")
