import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_core.prompts.prompt import PromptTemplate

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No OPENAI_API_KEY found in environment variables.")

# Initialize the embedding function and Chroma database
embedding_function = OpenAIEmbeddings(api_key=openai_api_key)
db = Chroma(persist_directory="chroma", embedding_function=embedding_function)

# Initialize OpenAI model for answer generation
llm = OpenAI(api_key=openai_api_key)

# Define custom prompt template
template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

# Initialize ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# Setup RetrievalQA with the retriever, custom prompt, and attach memory
retriever = db.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       memory=memory,
                                       return_source_documents=True)

def generate_response(query_text):
    llm_response = qa_chain.invoke(query_text)
    
    response_text = llm_response['result']
    sources = [source.metadata['source'] for source in llm_response["source_documents"]]
    return response_text, sources

def main():
    print("Walter, the literature chatbot, initialized. Type 'quit' to exit.")

    while True:
        query_text = input("You: ")
        if query_text.lower() in ['quit', 'exit']:
            print("Exiting Walter. Goodbye!")
            break

        response_text, sources = generate_response(query_text)
        
        # Format and print the response along with the sources
        formatted_response = f"Walter: {response_text}\nSources: {sources}"
        print(formatted_response)

if __name__ == "__main__":
    main()
