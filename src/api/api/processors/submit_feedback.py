from langsmith import Client


client = Client()

def submit_feedback(trace_id: str = None, feedback_score: int = None, feedback_text: str = "", feedback_source_type: str = "api"):

    # Only submit feedback if trace_id is provided
    if not trace_id:
        return
    
    if feedback_score is not None:
        client.create_feedback(
            run_id=trace_id,
            key="thumbs",
            score=feedback_score,
            feedback_source_type=feedback_source_type
        )

    if len(feedback_text) > 0:
        client.create_feedback(
            run_id=trace_id,
            key="comment",
            value=feedback_text,
            feedback_source_type=feedback_source_type
        )