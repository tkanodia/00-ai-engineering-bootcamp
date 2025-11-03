import openai

import instructor
from pydantic import BaseModel, Field
from langsmith import traceable, get_current_run_tree

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Document

import numpy as np

from api.api.models import RAGGenerationResponseWithReferences

from api.api.rag.utils.prompt_management import build_prompt_from_yaml, prompt_template_config

import cohere

@traceable(
    name="get_embeddings",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model": "text-embedding-3-small"},
)
def get_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        model=model,
        input=text
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            # embedding functions have no output / completion tokens
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }
    return response.data[0].embedding

@traceable(
    name="retrieve_data",
    run_type="retriever",
)
def retrieve_data(query, qdrant_client, k=5):
    """
    Retrieve k most similar items to the query from Qdrant collection.
    Using hybrid search with RRF fusion.
    """
    query_embedding = get_embeddings(query)
    response = qdrant_client.query_points(
        collection_name="Amazon-items-collection-hybrid-search",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="bm25",
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )


    retrieved_context_ids = [];
    # this is description of the product
    retrieved_context = [];
    similarity_scores = [];
    retrieved_content_ratings = []
    for point in response.points:   
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_context.append(point.payload["description"])
        similarity_scores.append(point.score)
        retrieved_content_ratings.append(point.payload["average_rating"])
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_content_ratings": retrieved_content_ratings
    }

@traceable(
    name="format_retrieved_context",
    run_type="prompt",
)
def format_retrieved_context(context_data):
    formatted_context = ""
    for id, context, rating in zip(context_data["retrieved_context_ids"], context_data["retrieved_context"], context_data["retrieved_content_ratings"]):
            formatted_context += f"- ID: {id}, description: {context}, rating: {rating}\n"
    return formatted_context

@traceable(
    name="create_prompt",
    run_type="prompt",
)
def create_prompt(query, preprocessed_retrieved_context):
    # processed_context = format_retrieved_context(preprocessed_retrieved_context)

    prompt = prompt_template_config("src/api/api/rag/prompts/retrieval_generation.yaml", "retrieval_generation")
    rendered_prompt = prompt.render(query=query, preprocessed_retrieved_context=preprocessed_retrieved_context)
    return rendered_prompt


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model": "gpt-4.1-mini"},
)
def generate_answer(prompt):
    client = instructor.from_openai(openai.OpenAI())
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        response_model=RAGGenerationResponseWithReferences
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return response

@traceable(
    name="rag_pipeline",
)
def rag_pipeline(query, qdrant_client, top_k=5):
    preprocessed_retrieved_context = retrieve_data(query, qdrant_client, top_k)
    prompt = create_prompt(query, preprocessed_retrieved_context)
    answer = generate_answer(prompt)

    final_response = {
        "answer": answer.answer,
        "references": answer.references,
        "question": query,
        "retrieved_context_ids": preprocessed_retrieved_context["retrieved_context_ids"],
        "retrieved_context": preprocessed_retrieved_context["retrieved_context"],
        "similarity_scores": preprocessed_retrieved_context["similarity_scores"]
    }

    return final_response

def rag_pipeline_wrapper(question, top_k=5):
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    result = rag_pipeline(question, qdrant_client, top_k) 
    used_context = []
    dummy_vector = np.zeros(1536).tolist()

    for item in result.get("references", []):
        payload = qdrant_client.query_points(
            collection_name="Amazon-items-collection-hybrid-search",
            query=dummy_vector,
            using="text-embedding-3-small",
            limit=1,
            with_payload=True,
            with_vectors=False,
            query_filter=Filter(
                must=[
                    FieldCondition(key="parent_asin", match=MatchValue(value=item.id))
                ]
            )
        ).points[0].payload
    
        image_url = payload.get("image", "")
        price = payload.get("price", "")
        if image_url not in [None, ""]:
            used_context.append({
                "image_url": image_url,
                "price": price,
                "description": item.description
            })

    return {
        "answer": result["answer"],
        "references": used_context
    }