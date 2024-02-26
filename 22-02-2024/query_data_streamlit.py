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

# Streamlit app layout
st.title("Walter, the Literature Chatbot")
st.sidebar.title("Conversation History")

# Function to update conversation history in a format that's consistent with the original application
def update_conversation_history(user_input, bot_response, sources):
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    st.session_state.conversation_history.append({"query": user_input, "response": bot_response, "sources": sources})

def get_conversation_history_as_list():
    if 'conversation_history' in st.session_state:
        return [{"You": entry["query"], "Walter": entry["response"]} for entry in st.session_state.conversation_history]
    return []

def get_latest_sources():
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        return st.session_state.conversation_history[-1]['sources']
    return []

# Function to create literature prompt
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

# Function to generate a response based on user input
def generate_response(query_text):
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        response_text = "I'm unable to find matching results in the literature for your query."
        sources = []
    else:
        sources = [doc.metadata.get("source", "Unknown source") for doc, _score in results]
        literature_prompt = create_literature_prompt(
            question=query_text,
            conversation_history=get_conversation_history_as_list(),
            sources=sources
        )
        response = model.invoke(literature_prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)
    
    # Update conversation history with the latest interaction
    update_conversation_history(query_text, response_text, sources)

    return response_text, sources

# Display conversation history in the sidebar
def display_conversation_history():
    for entry in get_conversation_history_as_list():
        st.sidebar.markdown(f"You: {entry['You']}")
        st.sidebar.markdown(f"Walter: {entry['Walter']}")

# Handling user input and generating responses
user_input = st.text_input("Talk to Walter", key="user_input")

def on_send():
    if user_input:  # Check if there is an input from the user
        response_text, sources = generate_response(user_input)
        # Display response in the main area
        st.write(f"Walter: {response_text}")
        if sources:
            st.write(f"Sources: {sources}")
        # Clear the input field after sending by using the callback mechanism
        st.session_state.user_input = ""

if st.button("Send", on_click=on_send):
    pass  # The button press is handled by the on_send callback function

# Always display the conversation history
display_conversation_history()

