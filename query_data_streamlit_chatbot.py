import streamlit as st
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

# Initialize the Chat model
model = ChatOpenAI(api_key=openai_api_key)

# Set a title for the app
st.title("Walter, the Literature Chatbot")

# Initialize chat history if it's not already in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to create a JSON-formatted prompt for the chatbot
def create_literature_prompt(question, conversation_history, sources):
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
    return json.dumps(prompt_structure, indent=2)

# Function to display chat history
def display_chat_history():
    for message in st.session_state.messages:
        author = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(author):
            st.markdown(message["content"])

# Function to generate response and update chat history
def generate_response_and_update_history(user_input):
    # Perform a similarity search in the Chroma database
    results = db.similarity_search_with_relevance_scores(user_input, k=3)
    
    # Check for no results or low relevance scores
    if len(results) == 0 or results[0][1] < 0.7:
        response_text = "I'm unable to find matching results in the literature for your query."
        sources = []
    else:
        sources = [doc.metadata.get("source", "Unknown source") for doc, _score in results]
        # Create a literature prompt with the provided context, question, and sources
        literature_prompt = create_literature_prompt(
            question=user_input,
            conversation_history=[{"role": message["role"], "content": message["content"]} for message in st.session_state['messages']],
            sources=sources
        )
        # Generate a response using the literature prompt
        response = model.invoke(literature_prompt)
        # Extract the text content from the response object
        response_text = response.text if hasattr(response, 'text') else str(response)
    
    # Update session state with the new messages
    st.session_state['messages'].append({"role": "user", "content": user_input})
    st.session_state['messages'].append({"role": "assistant", "content": response_text})
    display_chat_history()  # Make sure to display updated conversation history

# Callback function to handle send action
def handle_send():
    user_input = st.session_state.user_input
    if user_input:
        generate_response_and_update_history(user_input)
        # Clear the input box by setting it to an empty string
        st.session_state.user_input = ""

# Display existing chat history
display_chat_history()

# Chat input for the user with a callback on text submission
user_input = st.text_input("Ask Walter a question...", key="user_input", on_change=handle_send)

# Button to send the message, triggering the handle_send function
#st.button("Send", on_click=handle_send)