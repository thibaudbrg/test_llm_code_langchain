from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.load import dumps, loads

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



vectorstore = Chroma(persist_directory="data/chroma",
                     #collection_name="rag-chroma",
                     embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()



# RAG-Fusion: Related
template_fusion = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template_fusion)
model = ChatOpenAI()

generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

# RAG
template = """
You are an AI assistant expert in UBP's data, trained to support UBP employees with accurate and relevant information. When answering, consider the following:

- Use the comprehensive knowledge from UBP's documents to provide detailed and specific answers.
- If the answer is not directly available in the documents, use your understanding to offer insights or suggest where to find more information.
- Always aim to be helpful, clear, and concise. If uncertain , it's better to admit the limitation rather than providing potentially incorrect information.
- After providing an answer, list the sources from UBP's documents that were used to compile the response. If no specific document was used, mention it accordingly.

Context:
{context}

Given the context above, here's the question from a UBP employee:
Question:
{question}

Please craft your answer thoughtfully, bearing in mind the needs and expectations of UBP employees.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retrieval_chain_rag_fusion, "question": RunnablePassthrough()} 
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)