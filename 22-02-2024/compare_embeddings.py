from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from dotenv import load_dotenv
import os

def main():
    load_dotenv()

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("No OPENAI_API_KEY found in environment variables.")
    
    # Initialize the embedding function for OpenAI models
    embedding_function = OpenAIEmbeddings()

    # Get the embedding vector for the word 'apple'


#    vector = embedding_function.embed_query("apple")
#    # Print the embedding vector and its length
#    print(f"Vector for 'apple': {vector}")
#    print(f"Vector length: {len(vector)}")

#    # Load the evaluator for comparing embedding vectors
#    evaluator = load_evaluator("pairwise_embedding_distance")
#    # Words to be compared
#    words = ("apple", "iphone")
#    # Evaluate and print the distance between embeddings of the two words
#    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
#    print(f"Comparing ({words[0]}, {words[1]}): {x}")

# Entry point of the script
if __name__ == "__main__":
    main()