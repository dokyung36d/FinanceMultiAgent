from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

class AgentState(TypedDict):
    messages: list[Annotated[AnyMessage, add_messages]]


def generate(state: AgentState) -> AgentState:
    """
    `generate` 노드는 사용자의 질문을 받아서 응답을 생성하는 노드입니다.
    """
    messages = state['messages']
    ai_message = llm.invoke(messages)
    return {'messages': [ai_message]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node('generate', generate)

graph_builder.add_edge(START, 'generate')
graph_builder.add_edge('generate', END)


graph = graph_builder.compile()


query = "우리은행 대출상품을 알려줘"
initial_state = {'messages': [HumanMessage(query)]}
result = graph.invoke(initial_state)
print(result['messages'][-1].content)