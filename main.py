import os
from dotenv import load_dotenv

# Load environment variables FIRST before importing components that require API keys
load_dotenv()

from graph import compile_graph
from state import AgentState

def main():
    print("Welcome to the Multi-Agent RAG Chatbot!")
    print("Type 'exit' or 'quit' to stop.")
    
    # Needs API keys from environment
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set in the environment.")
        return

    # Compile the LangGraph
    app = compile_graph()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        if not user_input.strip():
            continue
            
        # Initialize starting state
        initial_state = AgentState(
            messages=[],
            question=user_input,
            next_agent="",
            documents=[],
            generation="",
            sql_result=""
        )
        
        # Execute graph. Invokes the state machine.
        print("\n--- THINKING ---")
        try:
            # invoke() returns the final state dictionary
            final_state = app.invoke(initial_state)
            
            # The final response is stored in 'generation'
            print(f"\nChatbot: {final_state.get('generation', 'No response generated.')}")
            
        except Exception as e:
            print(f"\n[Error]: An error occurred: {e}")

if __name__ == "__main__":
    main()
