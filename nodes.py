import os
from state import AgentState
from tools import get_web_search_tool, setup_and_get_retriever, setup_sqldb
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.agent_toolkits import create_sql_agent

# Initialize tools
web_search_tool = get_web_search_tool()
retriever = setup_and_get_retriever()
sql_db = setup_sqldb()

def router_node(state: AgentState):
    """
    Decides the next step (route) based on the user's question.
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    system_prompt = """You are an expert router directing the user's question to the most appropriate tool.
    You have three tools available:
    1. 'vector_search': Use this for questions about company policy, widget product launches, employee benefits, or HR.
    2. 'sql': Use this for questions about employee details, departments, or salaries.
    3. 'web_search': Use this for recent events, news, or general knowledge not covered by the other tools.
    4. 'llm': Use this for general conversational questions (e.g. "Hi", "How are you?").
    
    Return ONLY the exact single word identifying the tool from the options above: 'vector_search', 'sql', 'web_search', or 'llm'. No other text.
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    route = response.content.strip().lower()
    
    print(f"Routing to: {route}")
    return {"next_agent": route}

def retrieve_node(state: AgentState):
    """
    Retrieves documents based on the question.
    """
    print("---RETREIVE FROM VECTOR DB---")
    question = state["question"]
    documents = retriever.invoke(question)
    docs_text = [doc.page_content for doc in documents]
    return {"documents": docs_text}

def web_search_node(state: AgentState):
    """
    Performs a web search based on the question.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    result = web_search_tool.invoke(question)
    return {"documents": [result]}

def sql_node(state: AgentState):
    """
    Executes a SQL query using the SQL Agent.
    """
    print("---SQL AGENT---")
    question = state["question"]
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    agent_executor = create_sql_agent(llm, db=sql_db, agent_type="openai-tools", verbose=False)
    
    response = agent_executor.invoke({"input": question})
    return {"sql_result": response["output"]}

def generate_node(state: AgentState):
    """
    Synthesizes the final answer using retrieved context, web search results, or SQL results.
    """
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state.get("documents", [])
    sql_result = state.get("sql_result", "")
    route = state.get("next_agent", "llm")
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    if route == "sql":
        return {"generation": sql_result}
    elif route == "llm":
        prompt = f"Answer the user's question conversationally.\nQuestion: {question}"
        response = llm.invoke(prompt)
        return {"generation": response.content}
    else:
        # For vector_search and web_search, context is in documents
        context = "\n".join(documents)
        prompt = f"""You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        IMPORTANT: If the context contains an error message (such as a search rate limit), please inform the user about the error instead of answering the question.
        Context: {context}
        
        Question: {question}
        Answer:"""
        
        response = llm.invoke(prompt)
        return {"generation": response.content}
