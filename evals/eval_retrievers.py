from api.api.agent.retrieval_generation import rag_pipeline

from langsmith import Client
from qdrant_client import QdrantClient

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, IDBasedContextPrecision, IDBasedContextRecall, ResponseRelevancy

# Initialize LLM for ragas
llm = ChatOpenAI(model="gpt-4.1-mini")
ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

# run refers to the actual rag pipeline run / output
# example refers to the reference output in eval dataset for same input question

async def ragas_faithfulness(run, example):
    """
    Calculate the faithfulness score of the RAG pipeline.
    """

    sample = SingleTurnSample(
        user_input=run.outputs['question'],
        response=run.outputs['answer'],
        retrieved_contexts=run.outputs['retrieved_context'],
    )
    scorer = Faithfulness(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)

async def ragas_response_relevancy(run, example):
    """
    Calculate the response relevancy score of the RAG pipeline.
    """
    sample = SingleTurnSample(
        user_input=run.outputs['question'],
        response=run.outputs['answer'],
        retrieved_contexts=run.outputs['retrieved_context'],
    )
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    return await scorer.single_turn_ascore(sample)

async def ragas_id_based_context_precision(run, example):
    """
    Calculate the ID-based context precision score of the RAG pipeline.
    """
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs['retrieved_context_ids'],
        reference_context_ids=example.outputs['reference_context_ids'],
    )
    scorer = IDBasedContextPrecision()
    return await scorer.single_turn_ascore(sample)


async def ragas_id_based_context_recall(run, example):
    """
    Calculate the ID-based context recall score of the RAG pipeline.
    """
    sample = SingleTurnSample(
        retrieved_context_ids=run.outputs['retrieved_context_ids'],
        reference_context_ids=example.outputs['reference_context_ids'],
    )
    scorer = IDBasedContextRecall()
    return await scorer.single_turn_ascore(sample)


results = ls_client.evaluate(
    lambda x: rag_pipeline(x['question'], qdrant_client),
    data="rag-eval-dataset",
    evaluators=[
        ragas_faithfulness, 
        ragas_response_relevancy, 
        ragas_id_based_context_precision, 
        ragas_id_based_context_recall
    ],
    experiment_prefix="eval_retrievers",
)