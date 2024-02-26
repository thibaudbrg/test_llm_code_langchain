from rag_fusion.chain import chain, retriever



def test_chain():
    
    print(
        chain.invoke({"question": "What is alibi"})
    )
    #print(retriever)


#def test_retriever_directly():
#    # retriever.get_relevant_documents("what did he say about ketanji brown jackson"):
#
#
#    question_text = "What is the capital of France?"
#    context = retriever.invoke(question_text)
#
#    if context:
#        print("Retrieved context:", context)
#    else:
#        print("No context was retrieved.")
