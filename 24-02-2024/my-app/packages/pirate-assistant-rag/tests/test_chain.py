from pirate_assistant_rag.chain import chain
import langsmith
from datetime import datetime
from langchain import chat_models, smith


def test_chain():
    print(
        chain.invoke({"text": "What is best, gold or marriage"}).content
    )


#def test_chain2():
#    client = langsmith.Client()
#    chain_results = client.run_on_dataset(
#        dataset_name="my-app-dataset-test",
#        llm_or_chain_factory=chain,
#        project_name=f"my-app-test-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
#        concurrency_level=5,
#        verbose=True,
#    )
