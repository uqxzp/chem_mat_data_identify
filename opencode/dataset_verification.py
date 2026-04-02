"""
Verify whether papers release new molecular datasets.

The script reads titles from a JSONL file, sends batched verification prompts
to an LLM, and prints the results.

Example:
    PYTHONPATH=. python dataset_verification.py \
      --jsonl data/unlabeled_scored_flagged.jsonl \
      --batch_size 1

Note: currently, batch size 1 is recommended in case the LLM times out. 
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from opencode_client import send_message

DEFAULT_FLAGGED = Path("data/unlabeled_scored_flagged.jsonl")

# Externalize logic for better results

DATA_VERIFICATION_PROMPT = """
You are verifying whether a scientific paper releases a new or newly curated molecular dataset.

Given a paper title, first identify the exact paper. Then inspect the paper’s official page and there, check in particular:
1. Data Availability / Code Availability / Associated Content sections
2. Supporting Information / Supplementary Information
3. Any external repositories linked by the paper (e.g. Zenodo, Figshare, GitHub, etc.)

Decision rule:
- YES = the paper releases its own new or newly curated dataset, and you found concrete evidence of access to the data (supplementary file, dataset link, repository, downloadable files, or explicit data availability statement).
- NO = after checking the expected locations, you found no evidence that the paper releases a new dataset, or it only uses previously existing datasets without releasing a new/curated version.
- UNSURE = use only if the paper strongly suggests a dataset exists but the actual released data cannot be verified.

Important:
- For YES, cite the strongest evidence you found.
- Prefer evidence from the paper page, supplementary files, or linked repositories.
- A code repository alone is not enough unless it clearly contains or links to the dataset.
- Format Check: Data must be molecular data, e.g. SMILES, InChI, SELFIES, or 3D formats (XYZ, SDF, MOL, PDB, CIF, etc.).

Tool use:
- Do not ask for permissions or mention access limitations. You are allowed to fetch any needed pages.
- When reading any webpage, ALWAYS use `crawlfetch` (NOT `webfetch`).

Return exactly this format:

Input: <exact input title>
Best match: <full matched paper title>
Search trail: <1-2 short sentences saying where you checked>
Evidence: <best evidence link/DOI/file/repository, or "None found">
Verdict: <YES / NO / UNSURE>
Reason: <brief justification>

Final rule: take your time and research thoroughly. I value a 2-minute accurate response over a 5-second hallucination.

---

Paper title:
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
    if len(titles) > 1:
        numbered_titles = "\n".join(
            f"{idx}. {title}" for idx, title in enumerate(titles, start=1)
        )
    else:
        numbered_titles = titles[0]
    return DATA_VERIFICATION_PROMPT.format(titles=numbered_titles), numbered_titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_FLAGGED,
        help="Input JSONL file with papers to verify.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of papers checked per prompt."
    )
    args = parser.parse_args()

    titles = load_titles(args.jsonl)
    if not titles:
        raise SystemExit(f"No titles found in {args.jsonl}")

    for batch_index, batch in enumerate(chunked(titles, args.batch_size), start=1):
        prompt, titles = build_prompt(batch)
        print(f"Sending batch {batch_index} with {len(batch)} title(s):\n{titles}")
        try:
            response = send_message(prompt, 180)
        except Exception as exc:
            print(f"Batch {batch_index} failed: {exc}")
            print("Skipping to next batch.")
            print()
            continue
        header = f"# Batch {batch_index} ({len(batch)} titles)"
        print(header)
        print(response.strip())
        print()


if __name__ == "__main__":
    main()
