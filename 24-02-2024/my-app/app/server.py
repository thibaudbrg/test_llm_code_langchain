from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
#from pirate_speak.chain import chain as pirate_speak_chain
from pirate_assistant_rag.chain import chain as pirate_assistant_rag_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, pirate_assistant_rag_chain, path="/pirate_assistant_rag")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
