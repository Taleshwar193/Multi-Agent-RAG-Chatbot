from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import router_node, retrieve_node, web_search_node, sql_node, generate_node

def route_conditional(state: AgentState):
    """
    Function used in conditional edges to direct state to the correct node based on router decision.
    """
    return state["next_agent"]

def compile_graph():
    """
    Builds and compiles the orchestrator graph.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("sql_agent", sql_node)
    workflow.add_node("generate", generate_node)
    
    # Build graph
    # 1. Start is always the router
    workflow.set_entry_point("router")
    
    # 2. Add conditional edges from router to respective agents
    workflow.add_conditional_edges(
        "router",
        route_conditional,
        {
            "vector_search": "retrieve",
            "web_search": "web_search",
            "sql": "sql_agent",
            "llm": "generate"
        }
    )
    
    # 3. Add regular edges from agents to the generation node (except LLM and SQL which generate directly)
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("sql_agent", "generate")
    
    # 4. End edge
    workflow.add_edge("generate", END)
    
    return workflow.compile()
