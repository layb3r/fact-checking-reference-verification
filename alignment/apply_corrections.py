"""
Apply corrections from the audited dataset back to the original format.

For each entry where label_needs_correction is true,
replace claim_text with rewritten_claim and strip audit fields.

Usage:
    python alignment/apply_corrections.py
"""

import json
from pathlib import Path

ALIGNMENT_DIR = Path(__file__).resolve().parent
INPUT_PATH = ALIGNMENT_DIR / "data" / "negatives_added_over_claim_with_citation_audited.json"
OUTPUT_PATH = ALIGNMENT_DIR / "data" / "negatives_added_over_claim_with_citation_corrected.json"

AUDIT_FIELDS = {
    "llm_judged_label", "llm_reasoning", "llm_confidence",
    "label_needs_correction", "rewritten_claim", "rewrite_rationale",
    "claim_cleaned", "claim_cleaning_log", "extracted_sub_claim",
}


def main():
    if not INPUT_PATH.exists():
        print(f"Audited file not found: {INPUT_PATH}")
        print("Run dataset_label_audit.py first.")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        instances = json.load(f)

    total = len(instances)
    corrected = 0
    replaced = 0

    for inst in instances:
        needs_correction = inst.get("label_needs_correction", False)
        rewritten = inst.get("rewritten_claim", "")

        if needs_correction and rewritten:
            inst["claim_text"] = rewritten
            replaced += 1
            corrected += 1
        elif needs_correction and not rewritten:
            corrected += 1

        for field in AUDIT_FIELDS:
            inst.pop(field, None)

    print(f"Total entries: {total}")
    print(f"Entries needing correction: {corrected}")
    print(f"Claim texts replaced: {replaced}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(instances, f, indent=2, ensure_ascii=False)

    print(f"Corrected dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
