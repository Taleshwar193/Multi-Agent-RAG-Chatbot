import streamlit as st
import os
import asyncio
from dotenv import load_dotenv

# Ensure there's an event loop for async operations (like DuckDuckGo)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables FIRST
load_dotenv()

from graph import compile_graph
from state import AgentState

# Ensure page configuration is set
st.set_page_config(page_title="Multi-Agent RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Multi-Agent RAG Chatbot")
st.markdown("Ask me anything! I dynamically route your queries to **Vector Databases**, **SQL Databases**, or **Web Search** based on context.")

# Validate API Key
if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
    st.error("⚠️ OPENAI_API_KEY is not set or is invalid. Please update your `.env` file.")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "app" not in st.session_state:
    # Compile graph once and store in session state
    st.session_state.app = compile_graph()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Initialize starting state
                initial_state = AgentState(
                    messages=[],
                    question=prompt,
                    next_agent="",
                    documents=[],
                    generation="",
                    sql_result=""
                )
                
                # Execute graph
                final_state = st.session_state.app.invoke(initial_state)
                
                # The final response is stored in 'generation'
                response_text = final_state.get('generation', 'No response generated.')
                route_taken = final_state.get('next_agent', 'unknown')
                
                # Show where the router went
                st.info(f"Routed to: **{route_taken}**")
                st.markdown(response_text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
