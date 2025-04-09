import argparse
import json
import textwrap
from pathlib import Path
import importlib.resources
from path_utils import PROCESSED_DATA_PATH
from SearchApp.search_orchestrator import SearchOrchestrator

def main():
    """
    Command-line interface for running semantic search.
    """
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser.add_argument("--query", type=str, required=True, help="Enter a search query.")
    args = parser.parse_args()


    id_mapping_path = PROCESSED_DATA_PATH / "Production" / "id_mapping.json"
    with open(id_mapping_path, "r") as f:
        id_mapping = json.load(f)

    with importlib.resources.files(__package__).joinpath("production_config.json").open("r") as f:
        production_config = json.load(f)


    orchestrator = SearchOrchestrator(
        config=production_config,
        id_mapping=id_mapping
    )

    results = orchestrator.search(args.query)

    print(format_search_results(results))

def format_search_results(results):
    formatted_output = "\n **Top Results:**\n"
    
    for i, res in enumerate(results[:6], start=1):
        text = res["text"]
        score = res["score"]

        if score < .00001:
            continue

        clean_text = text.replace("\n", " ").strip()
        wrapped_text = textwrap.fill(clean_text, width=80)

        formatted_output += f"{i}. {wrapped_text}\n   (Score: {score:.4f})\n\n"

    return formatted_output

if __name__ == "__main__":
    main()
