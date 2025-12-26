import sys
import os
from pathlib import Path

# Setup project path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Project imports
from logger.custom_logger import CustomLogger
from config.settings_loader import load_config
from pipeline.embed_and_persist import create_embed_and_persist_service

# LangChain and LangSmith imports
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
from langchain_openai import AzureOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load configuration and logger
config = load_config("config/config.yaml")
logger = CustomLogger().get_logger()

# Initialize retriever and load vector store
retriever = create_embed_and_persist_service()
loaded_store = retriever.load_vector_store(config["retriever"]["vector_database_directory"])

# Initialize LLM for evaluation
llm = AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_LLM_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_LLM_MODEL_ENDPOINT"),
)

def correctness_evaluator(run: Run, example: Example) -> dict:
    """Evaluate if retrieved content matches the reference answer."""
    
    # Get inputs and outputs
    actual_output = run.outputs.get("answer", "")
    expected_output = example.outputs.get("answer", "")
    input_question = example.inputs.get("question", "")
    
    # Create evaluation prompt
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluator judging the quality of retrieved documents from a vector database.

Your evaluation must focus on whether the retrieved content semantically and factually covers the information present in the reference document needed to answer the query.

Do NOT judge style, wording, or formatting.
Judge only semantic coverage, factual alignment, and sufficiency.

Scoring guidelines:
- Relevance Score:
  - 10 = Retrieved content fully covers all key facts from the reference.
  - 5  = Retrieved content partially covers key facts but is incomplete.
  - 0  = Retrieved content does not cover the key facts or is incorrect.

- Support@K:
  - Yes = Retrieved content is sufficient to correctly answer the query.
  - No  = Retrieved content is insufficient to answer the query."""),
        ("human", """Query: {input}

Reference Answer:
{expected_output}

Retrieved Content:
{actual_output}

Evaluate the retrieved content against the reference.

Respond ONLY with this exact format:
RelevanceScore: [0 | 5 | 10]
Support@K: [Yes | No]
Reasoning: [1-2 sentences explaining the judgment]""")
    ])
    
    # Invoke LLM
    try:
        chain = eval_prompt | llm
        response = chain.invoke({
            "input": input_question,
            "expected_output": expected_output,
            "actual_output": actual_output
        })
        
        # Parse response
        content = response.content.strip()
        lines = content.split('\n')
        
        score = 0
        support = "No"
        reasoning = ""
        
        for line in lines:
            if line.startswith("RelevanceScore:"):
                score_str = line.split(":")[-1].strip()
                score = int(score_str) if score_str.isdigit() else 0
            elif line.startswith("Support@K:"):
                support = line.split(":")[-1].strip()
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[-1].strip()
        
        return {
            "key": "correctness",
            "score": score / 10.0,  # Normalize to 0-1
            "comment": f"Support@K: {support} | {reasoning}"
        }
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        return {
            "key": "correctness",
            "score": 0,
            "comment": f"Evaluation failed: {str(e)}"
        }


def retrieval_documents(inputs: dict) -> dict:
    """Retrieve relevant documents for a given question."""
    
    question = inputs.get("question", "").strip()
    
    if not question:
        return {"answer": ""}
    
    # Perform similarity search
    results = loaded_store.similarity_search(
        question, 
        k=config["retriever"]["top_k"]
    )
    
    # Combine results
    answer_parts = []
    for i, result in enumerate(results, 1):
        logger.info(f"Result {i}: {result.page_content}")
        answer_parts.append(result.page_content)
    
    answer = "\n\n".join(answer_parts)
    
    return {"answer": answer}


# Run evaluation
if __name__ == "__main__":
    dataset_name = "rag_dataset"
    
    experiment_results = evaluate(
        retrieval_documents,
        data=dataset_name,
        evaluators=[correctness_evaluator],
        experiment_prefix="RAG-correctness-eval",
        description="Evaluating RAG system with custom correctness evaluator",
        metadata={
            "variant": "RAG with FAISS",
            "evaluator": "custom_correctness_llm_judge",
            "model": "gemini-2.0-flash-exp",
            "chunk_size": config.get("chunking", {}).get("chunk_size", 1000),
            "chunk_overlap": config.get("chunking", {}).get("overlap", 200),
            "k": config["retriever"]["top_k"],
        },
    )
    
    print("\nEvaluation completed! Check the LangSmith UI for detailed results.")
