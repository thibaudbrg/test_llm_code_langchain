import argparse
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Path to the Chroma database directory
CHROMA_PATH = "chroma"

# Template for the prompt to be used with the Chat model
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Main function to execute the script
def main():
    # Load environment variables, particularly the OpenAI API key
    load_dotenv()

    # Retrieve the OpenAI API key from the environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    # Raise an error if the API key is not found
    if not openai_api_key:
        raise ValueError("No OPENAI_API_KEY found in environment variables.")

    # Set up command-line interface for input
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Initialize the embedding function and Chroma database
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform a similarity search in the Chroma database
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # Check for no results or low relevance scores
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    # Combine the context of the top results for the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # Create and format the prompt using the template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Initialize the Chat model and generate a response
    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    # Retrieve the sources of the documents used in the context
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    # Format and print the response along with the sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

# Entry point of the script
if __name__ == "__main__":
    main()
