# langchain_test_gpt

This repository contains code and resources for testing the GPT model with the Langchain library.

## Directory Structure

- `chroma/`: Contains the SQLite database `chroma.sqlite3` and a UUID-named directory.
- `data/`: Contains the data used for testing, such as `alice_in_wonderland.md`.
- `.env`: A file to set environment variables.
- `compare_embeddings.py`: A Python script to compare embeddings.
- `create_database.py`: A Python script to create a database.
- `query_data.py`: A Python script to query data.
- `requirements.txt`: A file containing a list of items to be installed using pip install.

## Setup

1. Clone the repository:

```sh
git clone https://github.com/yourusername/langchain_test_gpt.git
```
2. Navigate to the project directory:
```sh
cd langchain_test_gpt
```
3. Install the required packages:
```sh
pip install -r requirements.txt
```
4. Usage
To create the database, run:
```sh
python create_database.py
```
To query data, run:
```sh
python query_data.py
```
To compare embeddings, run:
```sh
python compare_embeddings.py
