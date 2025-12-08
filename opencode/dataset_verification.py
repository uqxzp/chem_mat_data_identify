from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from opencode_client import send_message

DEFAULT_FLAGGED = Path("data/production/scored_flagged.jsonl")

# TODO: discuss acceptable and unacceptable format (SMILES, InChI, SELFIES, 3D Coordinate files (XYZ, SDF, MOL, PDB, CIF))

DATASET_CURATION_PROMPT = """
You are an expert Data Curator for Chemistry and Material Science. Your goal is to assess scientific papers to determine if they release datasets suitable for training Graph Neural Networks (GNNs).

I will provide a batch of 4 paper titles. For each title, perform the following steps:

STEP 1: Identification
Identify the specific publication. The input titles are generally correct but may contain minor spelling errors.
- If you find the paper: Use the correct, full title in your output.
- If you absolutely cannot match the title to a real publication: Output "Publication not found" and stop for that entry.

STEP 2: Dataset Verification (Strict Criteria)
Analyze if the paper releases a molecular dataset. I am EXCLUSIVELY interested in datasets that provide molecular structures in machine-readable formats.
- Acceptable formats (Verdict: Yes): SMILES or 3D Coordinate files (XYZ)
- Unacceptable formats (Verdict: No): Datasets that are only images, PDF tables, pure text descriptions, or spectral data without accompanying structures.
- No dataset (Verdict: No): Papers that discuss theory or synthesis without releasing a structured dataset.

Rules for Verdicts:
1. YES: Only if a dataset is explicitly mentioned (e.g., in Supporting Information, GitHub, Zenodo, or a database). You MUST append the specific format found (e.g., "Yes, SMILES" or "Yes, XYZ").
2. NO: If no dataset exists, or if the dataset is not in the acceptable formats listed above.
3. UNSURE: This verdict is highly discouraged. Use this only if the paper seemingly claims a dataset exists but the format is ambiguous even after analysis.

STEP 3: Output Formatting
You must strictly follow this format for every entry. Do not add conversational filler.

Input: <Exact input title provided by user>
Best match: <Corrected Title of the publication OR "Publication not found">
Verdict: <Yes, [Format] / No / Unsure>

---
Here is the batch of titles to process:
{titles}
"""


def load_titles(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    titles = [r["title"] for r in records]
    return titles


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def build_prompt(titles: list[str]) -> str:
    numbered_titles = "\n".join(
        f"{idx}. {title}" for idx, title in enumerate(titles, start=1)
    )
    return DATASET_CURATION_PROMPT.format(titles=numbered_titles)


def main():
    parser = argparse.ArgumentParser(
        description="Verify flagged titles with an Opencode LLM."
    )
    parser.add_argument("--flagged", type=Path, default=DEFAULT_FLAGGED)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    titles = load_titles(args.flagged)
    if not titles:
        raise SystemExit(f"No titles found in {args.flagged}")

    for batch_index, batch in enumerate(chunked(titles, args.batch_size), start=1):
        prompt = build_prompt(batch)
        print(f"Sending batch {batch_index} with {len(batch)} titles...")
        response = send_message(prompt)
        header = f"# Batch {batch_index} ({len(batch)} titles)"
        print(header)
        print(response.strip())
        print()


if __name__ == "__main__":
    main()
