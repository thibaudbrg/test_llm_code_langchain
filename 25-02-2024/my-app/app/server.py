from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_chroma.chain import chain as rag_chroma_chain
from rag_conversation.chain import chain as rag_conversation_chain
from rag_multi_query.chain import chain as rag_multi_query_chain
from rag_fusion.chain import chain as rag_fusion_chain


app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chroma_chain, path="/rag-chroma")
add_routes(app, rag_conversation_chain, path="/rag-conversation")
add_routes(app, rag_multi_query_chain, path="/rag-multi-query")
add_routes(app, rag_fusion_chain, path="/rag-fusion")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
