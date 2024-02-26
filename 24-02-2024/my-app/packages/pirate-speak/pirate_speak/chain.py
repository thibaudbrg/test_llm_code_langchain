from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub



# _prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant who speaks like a pirate",
#         ),
#         ("human", "{text}"),
#     ]
# )


_model = ChatOpenAI()
_prompt = hub.pull("thibaudbrg/best-pirate-ever")
_model = ChatOpenAI()

# if you update this, you MUST also update ../pyproject.toml
# with the new `tool.langserve.export_attr`

chain = _prompt | _model
