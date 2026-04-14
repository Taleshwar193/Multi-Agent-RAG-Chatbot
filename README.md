# Multi-Agent RAG Chatbot

An intelligent Chatbot that leverages **LangGraph** to dynamically route user questions to the appropriate tool based on context.

## Architecture

This project incorporates a robust routing agent that decides between the following pathways:
1. **Vector DB Resolver (`ChromaDB`)**: Queries sample documents when asked about internal policies or product launches.
2. **SQL Agent Resolver (`SQLite`)**: Leverages Langchain's SQL Agent to fetch structured database elements (e.g. Employee salaries, departments).
3. **Web Search Resolver (`DuckDuckGo`)**: Consults DuckDuckGo for general, unhandled knowledge and current events.
4. **General LLM**: Reverts back to baseline conversational memory handling when the query does not require specific datasets.

Evaluation is heavily tracked via **LangSmith** and formally scored through **RAGAS** via the dedicated `evaluate.py` evaluation suite.

## Setup Instructions

1. Ensure Python 3.9+ is installed.
2. Create and activate a Virtual Environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install project dependencies.
   ```bash
   pip install -r requirements.txt
   ```
4. Configure required API Keys. Open `.env.example`, rename it to `.env`, and fill in the required keys.
   ```env
   OPENAI_API_KEY=your_openai_key
   LANGCHAIN_API_KEY=your_langchain_key # For LangSmith Tracing
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=multi_agent_rag_chatbot
   ```

## Running the Application

To interact with the chatbot in the terminal:
```bash
python main.py
```

To run the RAGAS evaluation suite (requires the `.env` keys):
```bash
python evaluate.py
```

<!-- Maintained documentation update for GitHub contribution tracking -->
