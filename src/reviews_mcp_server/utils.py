from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Document, FusionQuery, Filter, FieldCondition, MatchAny
from langsmith import traceable
import openai
from langsmith import get_current_run_tree


def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }

    return response.data[0].embedding

def retrieve_reviews_data(query, items_ids, k=5):

    query_embedding = get_embedding(query)

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-reviews",
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=items_ids
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )

    retrieved_reviews_context_ids = []
    retrieved_reviews_context = []
    similarity_scores = []

    for result in results.points:
        retrieved_reviews_context_ids.append(result.payload["parent_asin"])
        retrieved_reviews_context.append(result.payload["text"])
        similarity_scores.append(result.score)

    return {
        "retrieved_reviews_context_ids": retrieved_reviews_context_ids,
        "retrieved_reviews_context": retrieved_reviews_context,
        "similarity_scores": similarity_scores,
    }



### retreival tool

def process_reviews_context(context):

    formatted_context = ""

    for id, chunk in zip(context["retrieved_reviews_context_ids"], context["retrieved_reviews_context"]):
        formatted_context += f"- ID: {id}, text: {chunk}\n"

    return formatted_context
