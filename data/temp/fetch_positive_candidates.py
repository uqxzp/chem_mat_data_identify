import json
import os
from itertools import product

from utils.fetch_openalex import search_openalex

# Fetch promising papers using combinations of concepts and keywords. This + model -> ~1:2.5 TP to FP

OUTPUT_FILE = "data/production/positive_candidates.jsonl"
MAX_SAMPLES_PER_QUERY = 500 # Fetch up to 500 papers for each query combination
MAX_PAGES_PER_QUERY = 3 # Limit to 3 pages (at 200 per page = 600 results max)

# --- OpenAlex Concept IDs ---
# Using concepts is more reliable than keyword searches for broad domains.
CHEMISTRY_CONCEPT = "C185592680"
MATERIALS_SCIENCE_CONCEPT = "C192562407"
GNN_CONCEPT = "C2778504328"

# --- Query Groups for building search strings ---
# Combining one term from each group is a more robust strategy than a single complex query

# Group 1: Terms indicating a dataset is being provided or benchmarked
DATA_PROVISION_TERMS = [
    "benchmark dataset", "data set", "training data", 
    "supplementary data", "data available", "code available"
]

# Group 2: Terms for common graph/molecule data formats
DATA_FORMAT_TERMS = [
    "SMILES", "InChI", "MOL2", "PDB", "CIF", "XYZ format" # Currently contains more than just SMILES and XYZ
]

# Group 3: Terms related to GNNs (alternative to the concept) to catch papers not using the GNN concept
GNN_KEYWORDS = [
    '"graph neural network"', '"graph convolutional network"', 'GNN', 'GCN'
]


def fetch_positive_candidates():
    """
    Fetches works from OpenAlex that are likely to contain GNN datasets
    for chemistry and materials science.
    """
    processed_dois = set()
    all_results = []

    domain_concepts = [CHEMISTRY_CONCEPT, MATERIALS_SCIENCE_CONCEPT]

    # --- Strategy 1: Concept-based search ---
    # Search for papers tagged with Chem/Mat AND GNN concepts, then look for data provision or format keywords
    print("\n--- Running Strategy 1: Concept-based search ---")
    query_keywords = DATA_PROVISION_TERMS + DATA_FORMAT_TERMS
    search_query = " OR ".join(f'"{term}"' for term in query_keywords)
    
    results = search_openalex(
        search_query=search_query,
        concepts=domain_concepts + [GNN_CONCEPT],
        max_pages=MAX_PAGES_PER_QUERY * 2
    )
    for item in results:
        doi = item.get("doi")
        if doi and doi not in processed_dois:
            all_results.append(item)
            processed_dois.add(doi)

    # --- Strategy 2: Keyword combination search ---
    # Finds papers not perfectly tagged with concepts. Combines a GNN keyword with a data provision/format keyword
    print("\n--- Running Strategy 2: Keyword combination search ---")
    query_combinations = product(GNN_KEYWORDS, DATA_PROVISION_TERMS + DATA_FORMAT_TERMS)

    for gnn_term, data_term in query_combinations:
        # Create a boolean AND query
        search_query = f'"{gnn_term}" AND "{data_term}"'
        
        results = search_openalex(
            search_query=search_query,
            concepts=domain_concepts, # Filter by domain
            max_pages=MAX_PAGES_PER_QUERY
        )
        
        for item in results:
            doi = item.get("doi")
            if doi and doi not in processed_dois:
                all_results.append(item)
                processed_dois.add(doi)
        
        print(f"Found {len(all_results)} unique candidates so far.")

    print(f"\nTotal unique positive candidates found: {len(all_results)}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for entry in all_results:
            f.write(json.dumps(entry) + "\n")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_positive_candidates()

