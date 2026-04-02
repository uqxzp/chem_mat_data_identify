A binary classifier using TinyLlama + QLoRA for identifying scientific publications that likely release new molecular datasets based on their titles and abstracts.

1. Run `data/fetch_oa_in_order.py` to fetch publications from OpenAlex
2. Run `data/score_papers.py` to score papers with the SLM classifier
3. Run `opencode/dataset_verification.py` to verify papers with an LLM agent

Used opencode config agent for the LLM-based verification step:
```
"agent": {
    "dataset-verification": {
        "description": "Find the publications with the given titles and verify whether they release a molecular dataset.",
        "steps": 24,
        "permission": {
            "websearch_*": "allow",
            "crawlfetch": "allow",
            "webfetch": "deny",
            "file_downloader_*": "deny",
            "bash": "deny",
            "read": "deny",
            "glob": "deny",
            "grep": "deny",
            "list": "deny",
            "edit": "deny",
            "patch": "deny",
            "codesearch": "deny",
            "skill": "deny",
            "todowrite": "deny",
            "todoread": "deny",
            "question": "deny",
            "task": {
                "*": "deny"
            }
        }
    }
}
```