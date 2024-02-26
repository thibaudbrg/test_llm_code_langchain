from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import os
import shutil

# Paths for Chroma database and data directory
CHROMA_PATH = "chroma"
DATA_PATH = "data/"

# Function to split text into manageable chunks
def split_text(documents: 'list[Document]'):
    # Initialize the text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # For demonstration purposes, print out the content and metadata of the 11th chunk
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

# Function to save chunks of text into the Chroma vector store
def save_to_chroma(chunks: 'list[Document]'):
    # If the Chroma database directory exists, delete it to start fresh
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new Chroma database from the chunks of documents
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    # Persist the database to disk
    db.persist()
    # Notify the user of the successful save operation
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# Function to generate the Chroma database
def generate_data_store():
    # Load documents from the specified directory
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()

    # Split documents into chunks for database storage
    chunks = split_text(documents)

    # Save the chunks to the Chroma database
    save_to_chroma(chunks)

# Main function to execute the script
def main():
    # Load environment variables, particularly the OpenAI API key
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("No OPENAI_API_KEY found in environment variables.")
    
    # Generate the data store
    generate_data_store()

# Entry point of the script
if __name__ == "__main__":
    main()
