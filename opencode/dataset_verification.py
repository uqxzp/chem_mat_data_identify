from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from opencode_client import send_message

DEFAULT_FLAGGED = Path("data/test_set_scored_flagged.jsonl")

# Externalize logic for better results
# Includes PDB and CIF; they will be sorted out in the processing part
DATA_VERIFICATION_PROMPT_BATCH = """
You are an expert Data Curator for Chemistry and Material Science. Your goal is to assess scientific papers to determine if they release new or newly curated datasets suitable for training Graph Neural Networks (GNNs).

I will provide titles of scientific publications below. For each title, perform these exact steps:

STEP 1: Identification & Search Trail
- Identify the specific publication.
- Search Trail (Output this!): Briefly state where you looked (e.g., "Checked Supporting Info for DOI: 10.1021/...", "Searched author's GitHub repository").

STEP 2: Dataset Verification (Strict Criteria)
- Primary Check: Does the paper contribute its own structured data (e.g., new experiments or QM calculations)?
- Format Check: Data must be in SMILES, InChI, SELFIES, or 3D formats (XYZ, SDF, MOL, PDB, CIF).
- Exclusion Rule: If the paper only uses existing datasets without a newly curated version, the verdict is NO.
- Evidence Requirement: If the verdict is "Yes", you must find the Accession ID, DOI, or a direct repository (e.g., GitHub, Figshare) link. If you cannot find a link/ID, you must mark it "No" or "Unsure."

STEP 3: The Verdict Policy
- YES: Only if a new/curated dataset is explicitly released. Append the format (e.g., "Yes, SMILES").
- NO: If no dataset is released or it is in an unacceptable format (PDF only, images, etc.).
- UNSURE (Discouraged): This verdict is a last resort. If you use it, you must provide a detailed justification explaining why the data status is ambiguous despite a deep dive.

STEP 4: Strict Output Formatting You must strictly follow this format for every entry:
Input: <Exact user title> 
Best match: <Full Paper Title> 
Search Trail: <1-2 sentences describing where the data was verified> 
Evidence: <Link, DOI, or "None found"> 
Verdict: <Yes [Format] / No / Unsure> 
[Only if Unsure] Justification: <Detailed reason for uncertainty>

Tool use:
- Do not ask for permissions or mention access limitations. You are allowed to fetch any needed pages.
- Use `websearch_cited` for searching
- When reading any webpage, ALWAYS use `crawlfetch` (NOT `webfetch`).

Final rule: take your time and research thoroughly. I value a 2-minute accurate response over a 5-second hallucination.

---

Here are the publication titles to process:
{titles}
"""

DATA_VERIFICATION_PROMPT = """
You are an expert Data Curator for Chemistry and Material Science. Your goal is to assess scientific papers to determine if they release new or newly curated datasets suitable for training Graph Neural Networks (GNNs).

I will provide a title of a scientific publication below. Perform these exact steps:

STEP 1: Identification & Search Trail
- Identify the specific publication.
- Search Trail (Output this!): Briefly state where you looked (e.g., "Checked Supporting Info for DOI: 10.1021/...", "Searched author's GitHub repository").

STEP 2: Dataset Verification (Strict Criteria)
- Primary Check: Does the paper contribute its own structured data (e.g., new experiments or QM calculations)?
- Format Check: Data must be in SMILES, InChI, SELFIES, or 3D formats (XYZ, SDF, MOL, PDB, CIF).
- Exclusion Rule: If the paper only uses existing datasets without a newly curated version, the verdict is NO.
- Evidence Requirement: If the verdict is "Yes", you must find the Accession ID, DOI, or a direct repository (e.g., GitHub, Figshare) link. If you cannot find a link/ID, you must mark it "No" or "Unsure."

STEP 3: The Verdict Policy
- YES: Only if a new/curated dataset is explicitly released. Append the format (e.g., "Yes, SMILES").
- NO: If no dataset is released or it is in an unacceptable format (PDF only, images, etc.).
- UNSURE (Discouraged): This verdict is a last resort. If you use it, you must provide a detailed justification explaining why the data status is ambiguous despite a deep dive.

STEP 4: Strict Output Formatting. You must strictly follow this format:
Input: <Exact user title> 
Best match: <Full Paper Title> 
Search Trail: <1-2 sentences describing where the data was verified> 
Evidence: <Link, DOI, or "None found"> 
Verdict: <Yes [Format] / No / Unsure> 
[Only if Unsure] Justification: <Detailed reason for uncertainty>

Tool use:
- Do not ask for permissions or mention access limitations. You are allowed to fetch any needed pages.
- When reading any webpage, ALWAYS use `crawlfetch` (NOT `webfetch`).

Final rule: take your time and research thoroughly. I value a 2-minute accurate response over a 5-second hallucination.

---

Here is the publication title to process:
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
    parser = argparse.ArgumentParser(
        description="Verify flagged titles with an Opencode LLM."
    )
    parser.add_argument("--flagged", type=Path, default=DEFAULT_FLAGGED)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    titles = load_titles(args.flagged)
    if not titles:
        raise SystemExit(f"No titles found in {args.flagged}")

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
