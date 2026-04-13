from typing import Annotated, TypedDict, Sequence, Optional
import operator
from langchain_core.messages import BaseMessage


# We define the State of the graph
class AgentState(TypedDict):
    """
    The state of the agent.

    Attributes:
        messages: A list of messages (chat history / conversation).
        question: The actual string question asked by the user.
        next_agent: Where the router decides to send the request (e.g. 'web_search', 'vector_search', 'sql', 'llm').
        documents: A list of retrieved documents from tools.
        generation: The generated answer from the LLM.
        sql_result: The result from querying the SQL Database.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    next_agent: str
    documents: list[str]
    generation: str
    sql_result: str
