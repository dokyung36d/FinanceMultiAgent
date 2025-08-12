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
from langchain_community.tools import TavilySearchResults
from typing import Literal

from loan_tools import equal_principal_schedule, equal_payment_schedule, bullet_repayment_schedule

load_dotenv()


class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model='gpt-4o')
tools = [equal_principal_schedule, equal_payment_schedule, bullet_repayment_schedule]
llm_with_tools = llm.bind_tools(tools)
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


tavily_search_tool = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

def web_search(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 웹 검색을 수행합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 웹 검색 결과가 추가된 state를 반환합니다.
    """
    query = state['query']
    results = tavily_search_tool.invoke(query)

    return {'context': results}


doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")
def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelvant']:
    """
    주어진 state를 기반으로 문서의 관련성을 판단합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        Literal['relevant', 'irrelevant']: 문서가 관련성이 높으면 'relevant', 그렇지 않으면 'irrelevant'를 반환합니다.
    """
    query = state['query']
    context = state['context']

    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})

    if response['Score'] == 1:
        return 'relevant'
    
    return 'irrelvant'


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
    rag_chain = prompt | llm_with_tools   # RAG 프롬프트와 LLM을 연결하여 체인을 만듭니다.
    response = rag_chain.invoke({'question': query, 'context': context})  # 질문과 문맥을 사용하여 응답을 생성합니다.
    return {'answer': response}  # 생성된 응답을 포함한 state를 반환합니다.

def check_finance_related(state: AgentState) -> Literal["finance", "non_finance"]:
    """
    사용자의 query가 금융 관련인지 판별하여 'finance' 또는 'non_finance' 문자열을 반환.
    retrieval 이전 단계에서 호출하므로 context는 사용하지 않음.
    """
    query = state["query"]

    system_msg = (
        "You are a strict binary classifier. "
        "Return exactly one token: finance or non_finance."
        "Finance includes: banking, loans, deposits, interest rates, cards, investments, "
        "stocks, bonds, funds, FX, crypto (as financial asset), insurance, taxes on financial assets, "
        "financial regulations/compliance, personal finance, corporate finance."
        "Non-finance includes: weather, travel tips (non-cost), general tech, cooking, sports scores, etc."
    )
    user_msg = f"Query: {query}\nAnswer with only: finance or non_finance"

    resp = llm.invoke([("system", system_msg), ("user", user_msg)])
    label = (resp.content or "").strip().lower()

    if label.startswith("finance"):
        return "finance"
    else:
        # 종료 이유를 state에 기록
        state["answer"] = "해당 질문은 금융 관련이 아니어서 처리하지 않습니다."
        return "non_finance"


def notify_non_finance(state: AgentState) -> AgentState:
    reason = state.get("answer", "해당 질문은 금융 관련이 아니어서 처리하지 않습니다.")
    return {"answer": reason}


graph_builder = StateGraph(AgentState)
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('web_search', web_search)
graph_builder.add_node('check_doc_relevance', check_doc_relevance)
graph_builder.add_node('check_finance_related', check_finance_related)
graph_builder.add_node("notify_non_finance", notify_non_finance)

graph_builder.add_conditional_edges(
    START,
    check_finance_related,
    {
        "finance": "retrieve",       # 금융이면 벡터스토어 검색
        "non_finance": "notify_non_finance"  # 비금융이면 즉시 종료
    }
)
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelvant': 'web_search'
    }
)
# graph_builder.add_edge('rewrite', 'web_search')
graph_builder.add_edge('web_search', 'generate')
graph_builder.add_edge('generate', END)
graph_builder.add_edge('notify_non_finance', END)

graph = graph_builder.compile()

from pathlib import Path

# Mermaid 기반 PNG 이미지 바이트 생성
png_bytes = graph.get_graph().draw_mermaid_png()

# # 파일로 저장
# output_path = Path("graph_structure.png")
# with open(output_path, "wb") as f:
#     f.write(png_bytes)

# print(f"그래프 구조 PNG 저장 완료: {output_path}")


query = "원금 1천만원, 연 6%, 24개월 동안 원금균등으로 상환할 시 첫 달에 내는 금액이 얼마지"
initial_state = {'query': query}
result = graph.invoke(initial_state)
print(result['answer'])  # 생성된 응답을 출력합니다.