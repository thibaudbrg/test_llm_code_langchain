from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

from langchain_openai import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

vectorstore = Chroma("data/chroma", OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# _prompt = ChatPromptTemplate.from_messages(
#      [
#          (
#              "system",
#              "You are an expert in cybersecurity, you help everyone to aquiere more knowledge.",
#          ),
#          ("human", "{question}"),
#      ]
#  )


_prompt = hub.pull("thibaudbrg/best-cybersecurity-teacher")
_model = ChatOpenAI()

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`
#chain = _prompt | _model

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | _prompt
    | _model
    | StrOutputParser()
)