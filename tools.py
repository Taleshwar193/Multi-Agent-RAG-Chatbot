import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine, text

from langchain_core.tools import tool
from duckduckgo_search import DDGS

@tool("web_search")
def custom_duckduckgo_tool(query: str) -> str:
    """Searches the web using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        return f"Web search is currently rate-limited by DuckDuckGo. Please wait a moment and try again. (Error: {e})"

def get_web_search_tool():
    """Returns the custom DuckDuckGo Search tool."""
    return custom_duckduckgo_tool

# 2. Vector DB (Chroma) Setup and Retriever Tool
def setup_and_get_retriever():
    """
    Sets up a Chroma vector store with sample data and returns a retriever.
    This simulates loading real internal documents.
    """
    # Sample documents mimicking internal knowledge base
    sample_docs = [
        "Company policy requires all employees to request PTO at least 2 weeks in advance.",
        "The new 'Super Widget' product will be officially launched in Q3 2026.",
        "The standard reimbursement for internet stipends is $50 per month.",
        "HR can be reached at hr_team@company.com for any benefits inquiries."
    ]
    
    docs = [Document(page_content=text) for text in sample_docs]
    
    # Split text into chunks (if they were longer)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)
    
    # Initialize OpenAI Embeddings based VectorStore
    # We use a persisting directory so it doesn't rebuild every time (rebuilds if doesn't exist)
    persist_directory = "./chroma_db"
    embeddings = OpenAIEmbeddings()
    
    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
        vectorstore.persist()
        
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. SQL DB (SQLite) Setup
def setup_sqldb():
    """
    Sets up an in-memory or file-based SQLite database with sample data.
    Returns the SQLDatabase instance.
    """
    db_file = "sample.db"
    engine = create_engine(f"sqlite:///{db_file}")
    
    # Create a mock table and populate it if it doesn't exist
    if not os.path.exists(db_file):
        with engine.begin() as conn:
            conn.execute(text('''
                CREATE TABLE employees (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    department TEXT,
                    salary INTEGER
                )
            '''))
            conn.execute(text('''
                INSERT INTO employees (name, department, salary) VALUES 
                ('Alice', 'Engineering', 120000),
                ('Bob', 'Sales', 95000),
                ('Charlie', 'Engineering', 115000),
                ('Diana', 'HR', 85000)
            '''))
            
    return SQLDatabase(engine=engine)
