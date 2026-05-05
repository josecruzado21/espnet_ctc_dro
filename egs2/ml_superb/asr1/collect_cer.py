#!/usr/bin/env python3
"""Collect CER per language for each experiment, for dev and test splits."""

import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def extract_cer(result_txt: Path) -> float | None:
    """Return the overall Err (CER) from the Sum/Avg line of a result.txt."""
    for line in result_txt.read_text().splitlines():
        if "Sum/Avg" in line:
            # Line format: |   Sum/Avg  |  N  N  |  Corr  Sub  Del  Ins  Err  S.Err  |
            parts = [p.strip() for p in line.split("|") if p.strip()]
            # parts[0]=Sum/Avg, parts[1]=counts, parts[2]=metrics
            metrics = parts[2].split()
            # Corr Sub Del Ins Err S.Err
            return float(metrics[4])
    return None


def collect_results(base_dir: Path) -> list[dict]:
    rows = []
    # Find all per-language result.txt files
    for result_txt in sorted(base_dir.glob("exp*/*/decode_*/*/score_cer/independent/*/result.txt")):
        parts = result_txt.parts
        # Extract components from path relative to base_dir
        # Structure: base/expN/model/decode_X/split/score_cer/independent/lang/result.txt
        exp = parts[len(base_dir.parts)]
        model = parts[len(base_dir.parts) + 1]
        decode = parts[len(base_dir.parts) + 2]
        split = parts[len(base_dir.parts) + 3]
        lang = parts[len(base_dir.parts) + 6]

        # Normalize split name: strip "org/" prefix if nested
        if split == "org":
            # path is base/expN/model/decode_X/org/<split>/score_cer/...
            # re-find with deeper glob
            continue

        cer = extract_cer(result_txt)
        if cer is not None:
            rows.append({
                "exp": exp,
                "model": model,
                "decode": decode,
                "split": split,
                "lang": lang,
                "cer": cer,
            })

    # Also handle the org/<split> nesting (dev_1h is under org/)
    for result_txt in sorted(base_dir.glob("exp*/*/decode_*/org/*/score_cer/independent/*/result.txt")):
        p = result_txt.parts
        b = len(base_dir.parts)
        exp = p[b]
        model = p[b + 1]
        decode = p[b + 2]
        split = p[b + 4]   # skip "org"
        lang = p[b + 7]

        cer = extract_cer(result_txt)
        if cer is not None:
            rows.append({
                "exp": exp,
                "model": model,
                "decode": decode,
                "split": split,
                "lang": lang,
                "cer": cer,
            })

    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No results found.")
        return

    # Group by split
    splits = sorted({r["split"] for r in rows})
    exps = sorted({r["exp"] for r in rows})
    langs = sorted({r["lang"] for r in rows})

    for split in splits:
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue

        print(f"\n{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        # Header
        header = f"{'exp':<10}" + "".join(f"{lang:>10}" for lang in langs)
        print(header)
        print("-" * len(header))

        for exp in exps:
            exp_rows = {r["lang"]: r["cer"] for r in split_rows if r["exp"] == exp}
            if not exp_rows:
                continue
            row_str = f"{exp:<10}"
            for lang in langs:
                val = exp_rows.get(lang)
                row_str += f"{val:>10.1f}" if val is not None else f"{'N/A':>10}"
            print(row_str)


def main():
    rows = collect_results(BASE_DIR)
    print_table(rows)

    # Optionally write CSV
    if "--csv" in sys.argv:
        import csv, io
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=["exp", "split", "lang", "cer", "model", "decode"])
        writer.writeheader()
        writer.writerows(rows)
        print("\n--- CSV ---")
        print(out.getvalue())


if __name__ == "__main__":
    main()
