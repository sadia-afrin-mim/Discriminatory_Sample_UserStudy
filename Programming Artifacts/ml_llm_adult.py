"""
Themis-style Discrimination Testing on Adult Income Dataset
============================================================
- Trains a Random Forest on the UCI Adult Income dataset
- Uses OpenAI API to generate discriminatory test pairs (causal discrimination)
- Protected attribute: sex (Male vs Female), keeping all other features fixed
- Reports: success rate of discriminatory samples + total time

Requirements:
    pip install openai scikit-learn pandas numpy requests

Usage:
    export OPENAI_API_KEY="sk-..."
    python themis_llm_adult.py
"""

import os
import time
import json
import random
import warnings
import numpy as np
import pandas as pd
from openai import OpenAI

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. Load & train on Adult Income dataset
# ──────────────────────────────────────────────

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

FEATURE_COLS = [
    "age", "workclass", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country"
]

PROTECTED_ATTR = "sex"
PROTECTED_VALUES = ["Male", "Female"]


def load_adult_data():
    """Download Adult Income dataset from UCI or use cached copy."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    try:
        df = pd.read_csv(url, names=COLUMN_NAMES, sep=",\\s*", engine="python", na_values="?")
        print(f"[INFO] Loaded {len(df)} rows from UCI repository.")
    except Exception as e:
        print(f"[WARN] Could not fetch from UCI: {e}")
        print("[INFO] Generating synthetic adult-like data for demo...")
        df = generate_synthetic_adult(n=1000)
    return df


def generate_synthetic_adult(n=1000):
    """Fallback: generate a small synthetic dataset with the same schema."""
    random.seed(42)
    np.random.seed(42)
    workclasses  = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"]
    educations   = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Masters", "Doctorate"]
    maritals     = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed"]
    occupations  = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty"]
    relationships= ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
    races        = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    sexes        = ["Male", "Female"]
    countries    = ["United-States", "Cuba", "Jamaica", "India", "Mexico", "South"]

    rows = []
    for _ in range(n):
        sex = random.choice(sexes)
        age = random.randint(18, 90)
        edu_num = random.randint(1, 16)
        income = ">50K" if (age > 35 and edu_num > 12 and sex == "Male" and random.random() > 0.3) else "<=50K"
        rows.append({
            "age": age,
            "workclass": random.choice(workclasses),
            "fnlwgt": random.randint(10000, 999999),
            "education": random.choice(educations),
            "education_num": edu_num,
            "marital_status": random.choice(maritals),
            "occupation": random.choice(occupations),
            "relationship": random.choice(relationships),
            "race": random.choice(races),
            "sex": sex,
            "capital_gain": random.choice([0, 0, 0, 2407, 7688, 99999]),
            "capital_loss": random.choice([0, 0, 0, 1902, 2042]),
            "hours_per_week": random.randint(1, 99),
            "native_country": random.choice(countries),
            "income": income
        })
    return pd.DataFrame(rows)


def preprocess_and_train(df):
    """Encode categoricals, train Random Forest, return model + encoders."""
    df = df.dropna().copy()
    df["income"] = df["income"].str.strip().map(lambda x: 1 if ">50K" in x else 0)

    encoders = {}
    for col in CATEGORICAL_COLS:
        df[col] = df[col].str.strip()
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[FEATURE_COLS]
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"[INFO] Random Forest trained. Test accuracy: {acc:.2%}")

    return model, encoders, df


def predict(model, encoders, sample_dict):
    """
    Run model prediction on a single sample dict (raw string values).
    Returns 0 (<=50K) or 1 (>50K).
    """
    row = {}
    for col in FEATURE_COLS:
        val = sample_dict[col]
        if col in encoders:
            le = encoders[col]
            val_str = str(val).strip()
            if val_str in le.classes_:
                row[col] = le.transform([val_str])[0]
            else:
                row[col] = 0
        else:
            row[col] = val
    X = pd.DataFrame([row])[FEATURE_COLS]
    return int(model.predict(X)[0])


# ──────────────────────────────────────────────
# 2. OpenAI-based test pair generator
# ──────────────────────────────────────────────

def build_prompt(df_raw, n_pairs=5):
    """
    Build a prompt asking the LLM to generate causal discriminatory test pairs.
    Each pair: same features except sex (Male vs Female).
    """
    # Sample a few real rows to ground the LLM
    sample_rows = df_raw[FEATURE_COLS + ["income"]].dropna().sample(3, random_state=42).to_dict(orient="records")
    sample_str = json.dumps(sample_rows, indent=2)

    prompt = f"""
You are a fairness testing assistant for an ML model trained on the Adult Income dataset.

The model predicts whether a person earns >50K or <=50K per year.
The protected attribute is "sex" (Male or Female).

Your task: Generate {n_pairs} test pairs where ONLY the "sex" field differs (Male vs Female),
and all other fields remain IDENTICAL. These are called causal discrimination test pairs.

The feature fields are:
{FEATURE_COLS}

Here are some example real rows from the dataset for reference:
{sample_str}

Rules:
- Each pair must have two entries: one with sex="Male", one with sex="Female"
- All other fields must be IDENTICAL between the two entries
- Use realistic values (match the dataset's value ranges and categories)
- Numeric fields: age (18-90), education_num (1-16), capital_gain (0-99999), capital_loss (0-4356), hours_per_week (1-99)
- Valid workclass values: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay
- Valid education values: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
- Valid marital_status values: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
- Valid occupation values: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
- Valid relationship values: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
- Valid race values: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
- Valid native_country values: United-States, Cuba, Jamaica, India, Mexico, South, Honduras, England, Canada, Germany

Return ONLY a valid JSON array of pair objects, no explanation, no markdown fences.
Each pair object has this structure:
{{
  "pair_id": 1,
  "male": {{ ...all {len(FEATURE_COLS)} feature fields with sex="Male" }},
  "female": {{ ...all {len(FEATURE_COLS)} feature fields with sex="Female" }}
}}
"""
    return prompt.strip()


def generate_pairs_via_llm(client, df_raw, n_pairs=5):
    """Call OpenAI to generate discriminatory test pairs."""
    prompt = build_prompt(df_raw, n_pairs)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=3000
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    pairs = json.loads(raw)
    return pairs


# ──────────────────────────────────────────────
# 3. Evaluate pairs for discrimination
# ──────────────────────────────────────────────

def evaluate_pairs(pairs, model, encoders):
    """
    For each pair, run both male and female through the model.
    A pair is discriminatory if the predictions differ.
    """
    results = []
    for pair in pairs:
        male_sample   = pair["male"]
        female_sample = pair["female"]

        pred_male   = predict(model, encoders, male_sample)
        pred_female = predict(model, encoders, female_sample)

        discriminatory = (pred_male != pred_female)
        results.append({
            "pair_id":       pair.get("pair_id", "?"),
            "pred_male":     ">50K" if pred_male   == 1 else "<=50K",
            "pred_female":   ">50K" if pred_female == 1 else "<=50K",
            "discriminatory": discriminatory
        })
    return results


# ──────────────────────────────────────────────
# 4. Main runner — Themis-style loop
# ──────────────────────────────────────────────

def run_themis_llm(n_pairs_per_call=5, max_samples=50, conf_threshold=0.5):
    """
    Themis-style loop:
      - Repeatedly ask LLM to generate test pairs
      - Evaluate each pair against the Random Forest
      - Stop when enough samples collected
      - Report success rate + timing
    """
    print("\n" + "="*60)
    print("  Themis-LLM: Discrimination Testing on Adult Income")
    print("="*60)

    # --- Setup ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    # --- Load data & train model ---
    print("\n[STEP 1] Loading dataset and training Random Forest...")
    t_train_start = time.time()
    df_raw = load_adult_data()
    model, encoders, df_processed = preprocess_and_train(df_raw)
    t_train_end = time.time()
    print(f"[INFO] Training completed in {t_train_end - t_train_start:.2f}s")

    # --- Themis-style sampling loop ---
    print(f"\n[STEP 2] Generating & evaluating test pairs (target: {max_samples} pairs)...")
    print("-"*60)

    all_results = []
    total_pairs = 0
    total_discriminatory = 0
    first_disc_time = None

    t_loop_start = time.time()

    while total_pairs < max_samples:
        batch_num = (total_pairs // n_pairs_per_call) + 1
        print(f"\n[Batch {batch_num}] Requesting {n_pairs_per_call} pairs from LLM...")

        t_batch_start = time.time()
        try:
            pairs = generate_pairs_via_llm(client, df_raw, n_pairs=n_pairs_per_call)
        except json.JSONDecodeError as e:
            print(f"  [WARN] LLM returned invalid JSON, skipping batch: {e}")
            continue
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            break

        t_batch_llm = time.time()
        print(f"  LLM response received in {t_batch_llm - t_batch_start:.2f}s ({len(pairs)} pairs)")

        batch_results = evaluate_pairs(pairs, model, encoders)
        t_batch_eval = time.time()

        for r in batch_results:
            total_pairs += 1
            disc = r["discriminatory"]
            if disc:
                total_discriminatory += 1
                if first_disc_time is None:
                    first_disc_time = time.time() - t_loop_start

            status = "DISCRIMINATORY" if disc else "not discriminatory"
            print(f"  Pair {r['pair_id']:>2}: Male={r['pred_male']:<7} Female={r['pred_female']:<7} --> {status}")

            all_results.append({
                **r,
                "cumulative_total": total_pairs,
                "cumulative_disc":  total_discriminatory,
                "running_rate":     total_discriminatory / total_pairs
            })

        print(f"  Batch eval time: {t_batch_eval - t_batch_llm:.4f}s")

        if total_pairs >= max_samples:
            break

    t_loop_end = time.time()

    # --- Final report ---
    total_time   = t_loop_end - t_loop_start
    success_rate = total_discriminatory / total_pairs if total_pairs > 0 else 0

    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  Total pairs tested       : {total_pairs}")
    print(f"  Discriminatory pairs     : {total_discriminatory}")
    print(f"  Success rate             : {success_rate:.1%}")
    print(f"  Time to first disc. pair : {first_disc_time:.2f}s" if first_disc_time else "  Time to first disc. pair : Not found")
    print(f"  Total process time       : {total_time:.2f}s")
    print(f"  Model training time      : {t_train_end - t_train_start:.2f}s")
    print("="*60)

    return {
        "total_pairs":          total_pairs,
        "discriminatory_pairs": total_discriminatory,
        "success_rate":         success_rate,
        "first_disc_time":      first_disc_time,
        "total_time":           total_time,
        "training_time":        t_train_end - t_train_start,
        "results":              all_results
    }


# ──────────────────────────────────────────────
# 5. Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    results = run_themis_llm(
        n_pairs_per_call=5,   # how many pairs to ask LLM for per call
        max_samples=30,       # total pairs to test (increase for more thorough testing)
        conf_threshold=0.5    # reserved for future confidence-based stopping
    )

    # Save results to CSV
    df_results = pd.DataFrame(results["results"])
    df_results.to_csv("discrimination_results.csv", index=False)
    print(f"\n[INFO] Detailed results saved to discrimination_results.csv")
