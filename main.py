from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain import hub

load_dotenv()


class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model='gpt-4o')
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

vector_store = Chroma(
    embedding_function=embeddings,
    collection_name='document_collection',
    persist_directory='./document_collection'
)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

def retrieve(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.
    """
    query = state['query']  # state에서 사용자의 질문을 추출합니다.
    docs = retriever.invoke(query)  # 질문과 관련된 문서를 검색합니다.

    return {'context': docs}  # 검색된 문서를 포함한 state를 반환합니다.




def generate(state: AgentState) -> AgentState:
    """
    사용자의 질문과 검색된 문서를 기반으로 응답을 생성합니다.

    Args:
        state (AgentState): 사용자의 질문과 검색된 문서를 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 응답이 추가된 state를 반환합니다.
    """
    context = state['context']  # state에서 검색된 문서를 추출합니다.
    query = state['query']  # state에서 사용자의 질문을 추출합니다.
    rag_chain = prompt | llm  # RAG 프롬프트와 LLM을 연결하여 체인을 만듭니다.
    response = rag_chain.invoke({'question': query, 'context': context})  # 질문과 문맥을 사용하여 응답을 생성합니다.
    return {'answer': response}  # 생성된 응답을 포함한 state를 반환합니다.


graph_builder = StateGraph(AgentState)
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_edge('retrieve', 'generate')
graph_builder.add_edge('generate', END)


graph = graph_builder.compile()


query = "우리은행 대출할 때 유의해야 할 점들을 알려줘"
initial_state = {'query': query}
result = graph.invoke(initial_state)
print("hello world")