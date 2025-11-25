import yaml
from jinja2 import Template
from langsmith import Client

ls_client = Client()

def prompt_template_config(prompt_template_path, prompt_key, model="gpt-4.1"):
    """
    Load a prompt template from a YAML file.
    
    Args:
        prompt_template_path: Path to the YAML file
        prompt_key: Key to look up in the prompts section (usually the model name like 'gpt-4.1')
        model: Model name to use as the prompt key (default: 'gpt-4.1')
    
    Returns:
        Jinja2 Template object
    """
    with open(prompt_template_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    # Use model as the key if it exists, otherwise fallback to prompt_key
    if model in yaml_data["prompts"]:
        prompt = yaml_data["prompts"][model]
    elif prompt_key in yaml_data["prompts"]:
        prompt = yaml_data["prompts"][prompt_key]
    else:
        # Fallback to first available prompt
        prompt = list(yaml_data["prompts"].values())[0]
    
    template = Template(prompt)
    return template


def build_prompt_from_yaml(query, preprocessed_retrieved_context):
    
    prompt_template_path = "./prompts/retrieval_generation.yaml"
    prompt_key = "retrieval_generation"
    template = prompt_template_config(prompt_template_path, prompt_key)

    rendered_prompt = template.render(query=query, preprocessed_retrieved_context=preprocessed_retrieved_context)
    return rendered_prompt

def prompt_registry(prompt_key):
    template = ls_client.pull_prompt(prompt_key)
    template = Template(template.messages[0].prompt.template)
    return template