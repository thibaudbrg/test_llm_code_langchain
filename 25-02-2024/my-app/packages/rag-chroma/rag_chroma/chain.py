from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



vectorstore = Chroma(persist_directory="data/chroma",
                     #collection_name="rag-chroma",
                     embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

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
# moderate = OpenAIModerationChain() -- deprecated

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
    #| moderate -- because has been deprecated
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)