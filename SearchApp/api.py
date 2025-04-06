from fastapi import FastAPI
from SearchApp.search_orchestrator import SearchOrchestrator
# from SearchApp.constants import PROCESSED_DATA_DIR
from path_utils import PROCESSED_DATA_PATH
import json
from pathlib import Path
import importlib.resources

app = FastAPI()

# Load config and id mapping here
id_mapping_path = PROCESSED_DATA_PATH / "Production" / "id_mapping.json"
with open(id_mapping_path, "r") as f:
    id_mapping = json.load(f)

with importlib.resources.files("SearchApp").joinpath("production_config.json").open("r") as f:
    production_config = json.load(f)

orchestrator = SearchOrchestrator(config=production_config, id_mapping=id_mapping)

@app.get("/search")
def search(query: str):
    results = orchestrator.search(query)
    return {"query": query, "results": results[:5]}
