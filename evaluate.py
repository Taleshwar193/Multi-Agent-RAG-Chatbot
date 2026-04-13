import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from graph import compile_graph
from state import AgentState

def run_evaluation():
    """
    Evaluates the RAG system using RAGAS for Faithfulness and Answer Relevancy.
    """
    # Sample test questions covering distinct scenarios we designed the agent for
    test_questions = [
        "What is the standard reimbursement for internet stipends?", # Vector DB
        "When is the Super Widget launching?", # Vector DB
        "What is Bob's salary?", # SQL DB
        "Who works in the HR department?", # SQL DB
        "What is the current capital of France?", # Web Search or LLM
    ]
    
    app = compile_graph()
    
    questions = []
    answers = []
    contexts = []
    
    print("Generating answers for evaluation dataset...")
    for q in test_questions:
        print(f"Processing: {q}")
        
        initial_state = AgentState(
            messages=[], question=q, next_agent="", documents=[], generation="", sql_result=""
        )
        final_state = app.invoke(initial_state)
        
        questions.append(q)
        answers.append(final_state.get("generation", ""))
        
        # Ragas requires context as a list of strings
        # If SQL or LLM directly answers, context is empty, which may fail faithfulness metric.
        # Ragas is specifically designed for RAG (retrievers), so empty context for SQL is normal.
        docs = final_state.get("documents", [])
        contexts.append(docs if docs else ["No context retrieved (SQL or LLM)"])
        
    
    # Ragas requires 'question', 'answer', 'contexts'
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    
    dataset = Dataset.from_dict(data)
    
    print("\nRunning RAGAS evaluation...")
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings()
    
    # Execute RAGAS evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=llm,
        embeddings=embeddings
    )
    
    print("\n========= RAGAS EVALUATION METRICS =========")
    print(result)
    print("============================================")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set.")
    else:
        run_evaluation()
