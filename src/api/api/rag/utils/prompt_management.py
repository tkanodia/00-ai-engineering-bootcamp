import yaml
from jinja2 import Template
from langsmith import Client

ls_client = Client()

def prompt_template_config(prompt_template_path, prompt_key):
    with open(prompt_template_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    prompt = yaml_data["prompts"][prompt_key]
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