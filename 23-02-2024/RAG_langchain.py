# Install the required packages (Run this in your terminal or as a shell command)
#!pip install --upgrade --quiet langchain langchain-openai faiss-cpu tiktoken python-dotenv

# Import necessary libraries
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No OPENAI_API_KEY found in environment variables.")


# Initialize the vector store with FAISS
vectorstore = FAISS.from_texts(["harrison worked at kensho"], embedding=OpenAIEmbeddings(api_key=openai_api_key))
retriever = vectorstore.as_retriever()

# Define templates
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

# Basic RAG functionality
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(chain.invoke("where did harrison work?"))

# Conversational Retrieval Chain
# Define a function to combine documents
def _combine_documents(docs, document_prompt=PromptTemplate.from_template(template="{page_content}"), document_separator="\n\n"):
    doc_strings = [doc['page_content'] for doc in docs]  # Assuming docs is a list of dicts with 'page_content'
    return document_separator.join(doc_strings)

# Define inputs and context for conversational QA chain
_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | ChatPromptTemplate.from_template(template)
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

conversational_qa_chain = _inputs | _context | prompt | model | StrOutputParser()

# Example invocation
print(conversational_qa_chain.invoke(
    {
        "question": "where did harrison work?",
        "chat_history": [],
    }
))

# With Memory and Returning Source Documents
# Initialize memory
memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

# Load memory and prepare inputs
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)

# The final chain including memory loading and saving
final_chain = loaded_memory | conversational_qa_chain

# Example invocation with memory
inputs = {"question": "where did harrison work?"}
result = final_chain.invoke(inputs)
print(result)

# Save context to memory after invocation
memory.save_context(inputs, {"answer": result['answer'].content})

# Example of loading memory
print(memory.load_memory_variables({}))
