"""
Extract questions where the correct answer appeared at least once across all
agents and all rounds in the debate.

Usage:
    python evaluation/EDA/Gemma9B_Llama8B/extract_correct_appeared.py

Outputs:
    - Prints a summary to stdout
    - Saves a filtered CSV to the same folder as each input file
"""

import csv
import os
import glob

# ---------------------------------------------------------------------------
# Configuration — all CSVs in the Gemma9B_Llama8B folder
# ---------------------------------------------------------------------------
FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
)

INPUT_FILES = glob.glob(os.path.join(FOLDER, "metrics_log_*.csv"))


def extract_correct_appeared(input_csv: str) -> str:
    """
    Read a debate metrics CSV and return rows where the correct answer
    appeared at least once in any agent's answer across any round.

    A column is a round-agent answer column if its name matches the pattern
    'round_<N>_<AgentName>'.

    Args:
        input_csv: Path to the input CSV file.

    Returns:
        Path to the output CSV containing only the filtered rows.
    """
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Identify round-answer columns (e.g. round_0_Agent_Gemma)
        round_cols = [c for c in fieldnames if c.startswith("round_")]

        rows_appeared = []
        total = 0

        for row in reader:
            total += 1
            correct = row.get("correct_answer", "").strip()
            if not correct:
                continue

            # Check if the correct answer appears in any agent/round cell
            appeared = any(
                row[col].strip() == correct
                for col in round_cols
                if row[col].strip()  # skip empty / '?'
            )

            if appeared:
                rows_appeared.append(row)

    # Build output filename
    base = os.path.splitext(os.path.basename(input_csv))[0]
    output_csv = os.path.join(FOLDER, f"{base}_correct_appeared.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_appeared)

    pct = 100.0 * len(rows_appeared) / total if total > 0 else 0.0
    print(f"\nFile: {os.path.basename(input_csv)}")
    print(f"  Total questions        : {total}")
    print(f"  Correct ever appeared  : {len(rows_appeared)}  ({pct:.1f}%)")
    print(f"  NOT appeared           : {total - len(rows_appeared)}  ({100-pct:.1f}%)")
    print(f"  Saved to               : {os.path.basename(output_csv)}")

    return output_csv


def main():
    if not INPUT_FILES:
        print(f"No metrics_log_*.csv files found in:\n  {FOLDER}")
        return

    print(f"Found {len(INPUT_FILES)} file(s) to process:")
    for f in INPUT_FILES:
        print(f"  {os.path.basename(f)}")

    for input_csv in sorted(INPUT_FILES):
        extract_correct_appeared(input_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
