"""
Themis-style Discrimination Testing on a Rule-Based Loan Program
================================================================
- Uses a rule-based loan decision program (Loan.py style)
- Uses OpenAI API to generate discriminatory test pairs (causal discrimination)
- Protected attribute: sex (Male vs Female), keeping all other features fixed
- Reports: success rate of discriminatory samples + total time

Requirements:
    pip install openai

Usage:
    export OPENAI_API_KEY="sk-..."
    python themis_llm_rulebased.py
"""

import os
import time
import json
from openai import OpenAI

# ──────────────────────────────────────────────
# 1. Rule-based loan decision program (Loan.py)
# ──────────────────────────────────────────────

def decision_logic(sex, race, income):
    """
    Rule-based loan decision program.
    Returns 1 (approved) or 0 (denied).
    Mirrors the Themis benchmark Loan.py program.
    """
    if sex == 'male':
        return 1
    elif race != 'green':
        if income == '0...50000' or income == 'more':
            return 0
        else:
            return 1
    else:
        if income == '0...50000' or income == 'more':
            return 0
        else:
            return 1


def predict_rule_based(sample_dict):
    """Run rule-based prediction on a single sample dict."""
    sex    = sample_dict.get("sex", "female").lower()
    race   = sample_dict.get("race", "green").lower()
    income = sample_dict.get("income", "0...50000").lower()
    return decision_logic(sex, race, income)


# ──────────────────────────────────────────────
# 2. Feature space definition
# ──────────────────────────────────────────────

FEATURE_COLS      = ["sex", "race", "income"]
PROTECTED_ATTR    = "sex"
PROTECTED_VALUES  = ["male", "female"]

VALID_VALUES = {
    "sex":    ["male", "female"],
    "race":   ["green", "blue", "red", "yellow"],
    "income": ["0...50000", "more", "less"]
}


# ──────────────────────────────────────────────
# 3. OpenAI-based test pair generator
# ──────────────────────────────────────────────

def build_prompt(n_pairs=5):
    """
    Build a prompt asking the LLM to generate causal discriminatory
    test pairs for the rule-based loan program.
    Each pair: same features except sex (male vs female).
    """
    prompt = f"""
You are a fairness testing assistant for a rule-based loan decision program.

The program decides whether to approve a loan application based on three input fields:
- sex: the applicant's gender
- race: the applicant's race category
- income: the applicant's income bracket

The protected attribute is "sex".

Your task: Generate {n_pairs} test pairs where ONLY the "sex" field differs
(one entry with sex="male", one with sex="female"), and all other fields remain IDENTICAL.
These are called causal discrimination test pairs.

Valid values for each field:
- sex: ["male", "female"]
- race: ["green", "blue", "red", "yellow"]
- income: ["0...50000", "more", "less"]

Rules:
- Each pair must have exactly two entries: one with sex="male", one with sex="female"
- All fields other than "sex" must be IDENTICAL between the two entries
- Use only the valid values listed above

Return ONLY a valid JSON array of pair objects, no explanation, no markdown fences.
Each pair object has this structure:
{{
  "pair_id": 1,
  "male":   {{ "sex": "male",   "race": "<value>", "income": "<value>" }},
  "female": {{ "sex": "female", "race": "<value>", "income": "<value>" }}
}}
"""
    return prompt.strip()


def generate_pairs_via_llm(client, n_pairs=5):
    """Call OpenAI to generate discriminatory test pairs."""
    prompt = build_prompt(n_pairs)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    pairs = json.loads(raw)
    return pairs


# ──────────────────────────────────────────────
# 4. Evaluate pairs for discrimination
# ──────────────────────────────────────────────

def evaluate_pairs(pairs):
    """
    For each pair, run both male and female through the rule-based program.
    A pair is discriminatory if the predictions differ.
    """
    results = []
    for pair in pairs:
        male_sample   = pair["male"]
        female_sample = pair["female"]

        pred_male   = predict_rule_based(male_sample)
        pred_female = predict_rule_based(female_sample)

        discriminatory = (pred_male != pred_female)
        results.append({
            "pair_id":        pair.get("pair_id", "?"),
            "pred_male":      "Approved" if pred_male   == 1 else "Denied",
            "pred_female":    "Approved" if pred_female == 1 else "Denied",
            "discriminatory": discriminatory
        })
    return results


# ──────────────────────────────────────────────
# 5. Main runner — Themis-style loop
# ──────────────────────────────────────────────

def run_themis_llm(n_pairs_per_call=5, max_samples=30):
    """
    Themis-style loop:
      - Repeatedly ask LLM to generate test pairs
      - Evaluate each pair against the rule-based program
      - Stop when enough samples collected
      - Report success rate + timing
    """
    print("\n" + "="*60)
    print("  Themis-LLM: Discrimination Testing on Rule-Based Loan Program")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    print(f"\n[STEP] Generating & evaluating test pairs (target: {max_samples} pairs)...")
    print("-"*60)

    all_results          = []
    total_pairs          = 0
    total_discriminatory = 0
    first_disc_time      = None

    t_loop_start = time.time()

    while total_pairs < max_samples:
        batch_num = (total_pairs // n_pairs_per_call) + 1
        print(f"\n[Batch {batch_num}] Requesting {n_pairs_per_call} pairs from LLM...")

        t_batch_start = time.time()
        try:
            pairs = generate_pairs_via_llm(client, n_pairs=n_pairs_per_call)
        except json.JSONDecodeError as e:
            print(f"  [WARN] LLM returned invalid JSON, skipping batch: {e}")
            continue
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            break

        t_batch_llm = time.time()
        print(f"  LLM response received in {t_batch_llm - t_batch_start:.2f}s ({len(pairs)} pairs)")

        batch_results = evaluate_pairs(pairs)
        t_batch_eval  = time.time()

        for r in batch_results:
            total_pairs += 1
            disc = r["discriminatory"]
            if disc:
                total_discriminatory += 1
                if first_disc_time is None:
                    first_disc_time = time.time() - t_loop_start

            status = "DISCRIMINATORY" if disc else "not discriminatory"
            print(f"  Pair {r['pair_id']:>2}: Male={r['pred_male']:<10} Female={r['pred_female']:<10} --> {status}")

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
    print("="*60)

    return {
        "total_pairs":          total_pairs,
        "discriminatory_pairs": total_discriminatory,
        "success_rate":         success_rate,
        "first_disc_time":      first_disc_time,
        "total_time":           total_time,
        "results":              all_results
    }


# ──────────────────────────────────────────────
# 6. Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    results = run_themis_llm(
        n_pairs_per_call=5,
        max_samples=30
    )

    df_results = pd.DataFrame(results["results"])
    df_results.to_csv("discrimination_results_rulebased.csv", index=False)
    print(f"\n[INFO] Detailed results saved to discrimination_results_rulebased.csv")
