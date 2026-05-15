import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =========================
# 1) CLI: expected inputs (ONLY 5)
# =========================
EXPECTED_ORDER = [
    "Sex",
    "Race",
    "Age",
    "Education_Num",
    "Hours_per_week",
]

TARGET_COL = "Target"

# Themis/CLI should pass exactly 5 values
if len(sys.argv) != (len(EXPECTED_ORDER) + 1):
    print("0")  # safe default
    sys.exit(0)

raw_inputs = sys.argv[1:]
input_dict_raw = dict(zip(EXPECTED_ORDER, raw_inputs))

# =========================
# 2) Load and prepare data
# =========================
df = pd.read_csv("adult_dataset.csv")
df.dropna(inplace=True)

# Normalize target to 0/1
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
df[TARGET_COL] = df[TARGET_COL].replace({">50K.": ">50K", "<=50K.": "<=50K"})
df[TARGET_COL] = df[TARGET_COL].map({"<=50K": 0, ">50K": 1})
df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)

# Drop fnlwgt if present
if "fnlwgt" in df.columns:
    df = df.drop(columns=["fnlwgt"])

FEATURES = EXPECTED_ORDER

# Safety check
missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
if missing:
    print("0")
    sys.exit(0)

X = df[FEATURES].copy()
y = df[TARGET_COL].astype(int)

# Force numeric columns
for col in ["Age", "Education_Num", "Hours_per_week"]:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X.dropna(inplace=True)
y = y.loc[X.index]

# =========================
# 3) Encode categoricals
# =========================
categorical_cols = X.select_dtypes(include="object").columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = X[col].astype(str).str.strip()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# =========================
# 4) Train model (Random Forest)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(X_train, y_train)

# =========================
# 5) Helper for input normalization
# =========================
def coerce_to_known_category(raw_value: str, known_map: dict):
    raw_value = str(raw_value).strip()
    if raw_value in known_map:
        return known_map[raw_value]
    lower_map = {k.lower(): v for k, v in known_map.items()}
    if raw_value.lower() in lower_map:
        return lower_map[raw_value.lower()]
    return 0  # fallback

# =========================
# 6) Build input row from CLI
# =========================
input_vector = []
for feat in FEATURES:
    val = input_dict_raw.get(feat)

    if feat in categorical_cols:
        enc_val = coerce_to_known_category(val, label_encoders[feat])
        input_vector.append(enc_val)
    else:
        try:
            input_vector.append(float(val))
        except Exception:
            print("0")

            sys.exit(0)

input_df = pd.DataFrame([input_vector], columns=FEATURES)

# =========================
# 7) Predict and print 0/1 only (Themis-compatible)
# =========================
pred = int(model.predict(input_df)[0])
print("1" if pred == 1 else "0")