from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.load import dumps, loads
from operator import itemgetter

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



vectorstore = Chroma(persist_directory="data/chroma",
                     #collection_name="rag-chroma",
                     embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()



# Multi Query: Different Perspectives
template_multi_gen = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_multi_gen = ChatPromptTemplate.from_template(template_multi_gen)

generate_queries = (
    prompt_multi_gen
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

retrieval_chain = generate_queries | retriever.map() | get_unique_union


# RAG prompt
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
model = ChatOpenAI()

# RAG chain
chain = (
    RunnableParallel({"context": retrieval_chain, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)

