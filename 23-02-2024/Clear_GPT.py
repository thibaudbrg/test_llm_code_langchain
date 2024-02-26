import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA

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

# Setup RetrievalQA
retriever = db.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                       chain_type="stuff",
                       retriever=retriever,
                       return_source_documents=True)

# Conversation History Management
class ConversationContext:
    def __init__(self):
        self.history = []

    def update(self, user_query, bot_response, sources):
        self.history.append({"query": user_query, "response": bot_response, "sources": sources})

    def get_history_as_list(self):
        return [{"You": entry["query"], "Walter": entry["response"]} for entry in self.history]

    def get_latest_sources(self):
        if self.history:
            return self.history[-1]['sources']
        return []

# Function to generate a response based on a query
def generate_response(query_text, conversation_context):
    llm_response = qa_chain.invoke(query_text)
    response_text = llm_response['result']
    sources = [source.metadata['source'] for source in llm_response["source_documents"]]
    return response_text, sources

def main():
    print("Walter, the literature chatbot, initialized. Type 'quit' to exit.")
    conversation_context = ConversationContext()

    while True:
        query_text = input("You: ")
        if query_text.lower() in ['quit', 'exit']:
            print("Exiting Walter. Goodbye!")
            break

        response_text, sources = generate_response(query_text, conversation_context)
        
        # Update conversation history
        conversation_context.update(query_text, response_text, sources)
        
        # Format and print the response along with the sources
        formatted_response = f"Walter: {response_text}\nSources: {sources}"
        print(formatted_response)

if __name__ == "__main__":
    main()
