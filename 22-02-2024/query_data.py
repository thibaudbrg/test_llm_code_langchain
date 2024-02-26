import argparse
import os
import json
from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables, particularly the OpenAI API key
load_dotenv()

# Retrieve the OpenAI API key from the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No OPENAI_API_KEY found in environment variables.")

# Initialize the embedding function and Chroma database
embedding_function = OpenAIEmbeddings(api_key=openai_api_key)
db = Chroma(persist_directory="chroma", embedding_function=embedding_function)

# Initialize the Chat model with a specific model version for 'Walter'
#model_version = "gpt-3.5-turbo"  # Example for GPT-3.5
#model = ChatOpenAI(api_key=openai_api_key, model=model_version)
model = ChatOpenAI(api_key=openai_api_key)

# Conversation History Management
class ConversationContext:
    def __init__(self):
        self.history = []

    def update(self, user_query, bot_response, sources):
        self.history.append({"query": user_query, "response": bot_response, "sources": sources})

    def get_history_as_list(self):
        # Convert conversation history into a list format
        return [{"You": entry["query"], "Walter": entry["response"]} for entry in self.history]

    def get_latest_sources(self):
        # Retrieve sources from the latest entry if available
        if self.history:
            return self.history[-1]['sources']
        return []

def create_literature_prompt(question, conversation_history, sources):
    """
    Creates a JSON-formatted prompt for a literature-specialized chatbot named Walter.
    """
    prompt_structure = {
        "role": "system",
        "content": f"Walter, a chatbot specialized in literature, is assisting a humanities student with their inquiry: {question}. "
                   f"Below is the conversation history and relevant literary sources."
                   "\n\nAnswer the student's question using ONLY information from the sources or past conversations. "
                   "Any additional information helping to answer the question should be properly referenced."
                   "\n\nHere is an example of how you should reference sources:"
                   "\n- Information from source [source1]."
                   "\n- Information from past conversation [conversation].",
        "sources": sources,
        "conversation": conversation_history,
        "question": question
    }




## Main function to execute the chatbot
#    """
#    Creates a structured prompt for a chatbot named Walter that specializes in literature,
#    to assist humanities students with their inquiries.
#    
#    Parameters:
#    - question (str): The user's question about literature.
#    - conversation_history (list): A list of previous interactions in the conversation.
#    - sources (list): A list of sources that Walter can reference for providing answers.
#    
#    Returns:
#    - str: A JSON-formatted prompt for the chatbot.
#    """
#    prompt_structure = {
#        "role": "system",
#        "content": ("You are helping me, a humanities student focusing on literature, with my questions. "
#                    "Below you'll find the conversation history and literary sources relevant to my queries "
#                    "(they are formatted appropriately)."
#                    "\n\nAnswer my question using ONLY facts from the sources or past conversation. "
#                    "Information that helps answer the question can also be added."
#                    "\n\nIf not specified, format the answer using an introduction followed by a list of bullet points. "
#                    "The facts you add should ALWAYS STRICTLY reference each fact you use. "
#                    "A fact is preferably referenced by ONLY ONE source (e.g. [sourceX]). "
#                    "If you use facts from past conversation, use [conversation] as a reference."
#                    "\n\nHere is an example on how to reference sources (referenced facts must STRICTLY match the source number):"
#                    "\n- Some information retrieved from source [sourceX]."
#                    "\n- Some information retrieved from source [sourceY] and some information retrieved from [conversation]."),
#        "sources": sources,
#        "conversation": conversation_history,
#        "question": question
#    }
    return json.dumps(prompt_structure, indent=2)


def main():
    print("Walter, the literature chatbot, initialized. Type 'quit' to exit.")
    conversation_context = ConversationContext()

    while True:
        query_text = input("You: ")
        if query_text.lower() in ['quit', 'exit']:
            print("Exiting Walter. Goodbye!")
            break

        # Perform a similarity search in the Chroma database
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        
        # Check for no results or low relevance scores
        if len(results) == 0 or results[0][1] < 0.7:
            response_text = "I'm unable to find matching results in the literature for your query."
            sources = []
        else:
            sources = [doc.metadata.get("source", "Unknown source") for doc, _score in results]
            # Create a literature prompt with the provided context, question, and sources
            literature_prompt = create_literature_prompt(
                question=query_text,
                conversation_history=conversation_context.get_history_as_list(),
                sources=sources
            )
            # Generate a response using the literature prompt
            response = model.invoke(literature_prompt)

            # Extract the text content from the response object
            response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Update conversation history
        conversation_context.update(query_text, response_text, sources)

        # Format and print the response along with the sources
        formatted_response = f"Walter: {response_text}\nSources: {sources}"
        print(formatted_response)

if __name__ == "__main__":
    main()