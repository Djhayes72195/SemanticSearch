from fastapi import FastAPI
from SearchApp.search_orchestrator import SearchOrchestrator

app = FastAPI()
orchestrator = SearchOrchestrator(dataset_name="SQuAD", processed_corpus_id="some_id", config={})

@app.get("/search")
def search(query: str):
    results = orchestrator.search(query)
    return {"query": query, "results": results[:5]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
