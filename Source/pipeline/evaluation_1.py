import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import os
import pandas as pd
from datetime import datetime
from logger.custom_logger import CustomLogger
from exceptions.custom_exception import DocumentPortalException
from config.settings_loader import load_config
from pipeline.embed_and_persist import create_embed_and_persist_service

from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

logger = CustomLogger().get_logger()
config = load_config("config/config.yaml")

# Initialize retriever and loaded_store as None - will be loaded when needed
_retriever = None
_loaded_store = None

def _get_loaded_store():
    """Lazy load the vector store only when needed."""
    global _retriever, _loaded_store
    if _loaded_store is None:
        _retriever = create_embed_and_persist_service()
        _loaded_store = _retriever.load_vector_store(config["retriever"]["vector_database_directory"])
        logger.info("Vector store loaded successfully")
    return _loaded_store

def get_result(input: str):
    try:
        logger.info("initializing retrieval pipeline")
        
        # Get the loaded store (lazy loading)
        loaded_store = _get_loaded_store()

        # Perform a similarity search
        results = loaded_store.similarity_search(input, k=config["retriever"]["top_k"])
        output = ""
        for i, result in enumerate(results):
            output += f"Document {i+1}:\n{result.page_content}\n\n"
        return output
    except Exception as e:
        logger.error(f"Error in RAG_pipeline initialization(retrieval): {str(e)}")
        raise DocumentPortalException(e, sys) from e
    
    
llm = AzureChatOpenAI(
    api_version=os.getenv("AZURE_OPENAI_LLM_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_LLM_MODEL_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
)


class RetrieverEvalResult(BaseModel):
    relevance_score: Literal[0, 5, 10] = Field(
        description="0 = no coverage, 0.5 = partial coverage, 1 = full coverage"
    )
    support_at_k: Literal["Yes", "No"] = Field(
        description="Whether retrieved content is sufficient to answer the query"
    )
    reasoning: str = Field(
        description="Brief justification (1–2 sentences)"
    )


parser = PydanticOutputParser(pydantic_object=RetrieverEvalResult)

eval_prompt = PromptTemplate(
    template="""You are an evaluator judging the quality of retrieved documents from a vector database.

Judge only semantic and factual coverage of the reference document needed to answer the query.
Do not judge style, wording, or formatting.

Scoring rules:
- relevance_score:
  - 10  = full coverage of key facts
  - 5   = partial coverage
  - 0   = no coverage or incorrect
- support_at_k:
  - Yes = sufficient to answer correctly without guessing
  - No  = insufficient to answer

{format_instructions}

Query:
{query}

Reference Document:
{reference}

Retrieved Content:
{retrieved}

Provide your evaluation:""",
    input_variables=["query", "reference", "retrieved"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = eval_prompt | llm | parser

def evaluate():
    try:
        logger.info("Starting evaluation process")
        
        # Read evaluation dataset
        eval_dataset_path = Path(project_root) / "Data" / "ground_truth" / "evaluation_dataset.csv"
        df = pd.read_csv(eval_dataset_path)
        logger.info(f"Loaded {len(df)} questions from evaluation dataset")
        
        # Prepare results list
        evaluation_results = []
        
        # Process each question
        for idx, row in df.iterrows():
            question = row['question']
            expected_answer = row['answer']
            
            logger.info(f"Processing question {idx + 1}/{len(df)}: {question[:50]}...")
            
            # Get retrieved documents
            retrieved_output = get_result(question)
            
            # Evaluate using LLM
            result = chain.invoke({
                "query": question,
                "reference": expected_answer,
                "retrieved": retrieved_output
            })
            
            # Store evaluation results
            evaluation_results.append({
                "question": question,
                "expected_answer": expected_answer,
                "retrieved_output": retrieved_output,
                "relevance_score": result.relevance_score,
                "support_at_k": result.support_at_k,
                "reasoning": result.reasoning
            })
            
            logger.info(f"Question {idx + 1} evaluated - Score: {result.relevance_score}, Support: {result.support_at_k}")
        
        # Create evaluation results DataFrame
        results_df = pd.DataFrame(evaluation_results)
        
        # Save to CSV with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(project_root) / "Data" / "evaluation" / f"size_{config['data_chunking']["recursive_text_splitter"]["chunk_size"]}_overlap_{config['data_chunking']["recursive_text_splitter"]["chunk_overlap"]}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Evaluation results saved to {output_path}")
        
        # Calculate and log summary statistics
        avg_relevance = results_df['relevance_score'].mean()
        support_count = (results_df['support_at_k'] == 'Yes').sum()
        logger.info(f"\nEvaluation Summary:")
        logger.info(f"Average Relevance Score: {avg_relevance:.2f}")
        logger.info(f"Support at K: {support_count}/{len(df)} ({support_count/len(df)*100:.1f}%)")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise DocumentPortalException(e, sys) from e


if __name__ == "__main__":
    evaluate()