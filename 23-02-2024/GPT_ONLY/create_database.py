from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import PyPDF2
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/"

def load_pdf_documents(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            try:
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            documents.append(Document(page_content=text, metadata={"source": filename}))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return documents

def load_and_process_documents(data_path, file_types):
    documents = []
    loaders = {
        '.md': DirectoryLoader(data_path, glob="*.md"),
        '.txt': DirectoryLoader(data_path, glob="*.txt"),
        '.html': DirectoryLoader(data_path, glob="*.html", loader_cls=UnstructuredHTMLLoader),
    }
    for ext, loader in loaders.items():
        if ext in file_types:
            docs = loader.load()
            documents.extend(docs)
    if '.pdf' in file_types:
        documents.extend(load_pdf_documents(data_path))
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
    file_types = ['.md', '.txt', '.html', '.pdf']
    documents = load_and_process_documents(DATA_PATH, file_types)
    chunks = split_text(documents)
    save_to_chroma(chunks)

def main():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("No OPENAI_API_KEY found in environment variables.")
    generate_data_store()

if __name__ == "__main__":
    main()
