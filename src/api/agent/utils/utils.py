from typing import Dict, Any, List, Optional, Type
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import ast
import inspect
import instructor
from litellm import completion
from api.agent.utils.prompt_management import prompt_template_config

def filter_messages_for_coordinator(messages):
    """
    Filter conversation messages for the coordinator agent.
    The coordinator doesn't use tools, so we need to remove tool-related messages
    or convert AI messages with tool_calls to simple text messages.
    
    This prevents OpenAI API errors about tool_calls without responses.
    """
    filtered_messages = []
    
    for message in messages:
        # Keep user/human messages as-is
        if isinstance(message, dict) and message.get("role") == "user":
            filtered_messages.append(message)
        elif hasattr(message, "__class__") and message.__class__.__name__ == "HumanMessage":
            filtered_messages.append(message)
        # Skip tool messages entirely
        elif isinstance(message, dict) and message.get("role") == "tool":
            continue
        elif hasattr(message, "__class__") and message.__class__.__name__ == "ToolMessage":
            continue
        # For assistant messages, strip tool_calls
        elif isinstance(message, dict) and message.get("role") == "assistant":
            # Create a clean assistant message without tool_calls
            clean_message = {
                "role": "assistant",
                "content": message.get("content", "")
            }
            # Only add if there's actual content
            if clean_message["content"]:
                filtered_messages.append(clean_message)
        elif hasattr(message, "__class__") and message.__class__.__name__ == "AIMessage":
            # Convert AIMessage to dict without tool_calls
            if message.content:
                filtered_messages.append({
                    "role": "assistant",
                    "content": message.content
                })
        else:
            # Keep other message types
            filtered_messages.append(message)
    
    return filtered_messages

def format_ai_message(response):
    """Format the AI response into a LangChain AIMessage object"""
    if response.tool_calls:
        tool_calls = []
        for i, tc in enumerate(response.tool_calls):
            # Sanitize tool name to only allow alphanumeric, underscore, and hyphen
            # This prevents OpenAI API errors for invalid function names
            sanitized_name = ''.join(c for c in tc.name if c.isalnum() or c in ('_', '-'))
            
            tool_calls.append({
                "id": f"call_{i}",
                "name": sanitized_name,
                "args": tc.arguments
            })
        
        ai_message = AIMessage(
            content=response.answer,
            tool_calls=tool_calls
        )
    else:
        ai_message = AIMessage(
            content=response.answer,
        )
    
    return ai_message


def get_type_from_annotation(annotation) -> str:
    """Convert a type annotation to a string representation suitable for tool descriptions."""
    if annotation is None:
        return "string"
    
    # If it's an AST node, unparse it to string
    if isinstance(annotation, ast.AST):
        try:
            annotation_str = ast.unparse(annotation)
        except:
            return "string"
    else:
        annotation_str = str(annotation)
    
    # Map Python types to JSON schema types (for OpenAI function calling)
    type_mapping = {
        'str': 'string',
        'int': 'integer',
        'float': 'number',
        'bool': 'boolean',
        'list': 'array',
        'dict': 'object',
        'List': 'array',
        'Dict': 'object',
        'Any': 'string',
        'None': 'null',
    }
    
    # Handle simple types
    if annotation_str in type_mapping:
        return type_mapping[annotation_str]
    
    # Handle Optional types (e.g., Optional[str])
    if 'Optional' in annotation_str or 'Union' in annotation_str:
        # Extract the non-None type
        for type_name in type_mapping.keys():
            if type_name in annotation_str:
                return type_mapping.get(type_name, 'string')
    
    # Handle generic types like List[str], Dict[str, Any]
    if annotation_str.startswith(('List[', 'list[')):
        return 'array'
    elif annotation_str.startswith(('Dict[', 'dict[')):
        return 'object'
    
    # Default to string for unknown types
    return 'string'

def parse_docstring_params(docstring: str) -> Dict[str, str]:
    """Extract parameter descriptions from docstring (handles both Args: and Parameters: formats)."""
    params = {}
    lines = docstring.split('\n')
    in_params = False
    current_param = None
    
    for line in lines:
        stripped = line.strip()
        
        # Check for parameter section start
        if stripped in ['Args:', 'Arguments:', 'Parameters:', 'Params:']:
            in_params = True
            current_param = None
        elif stripped.startswith('Returns:') or stripped.startswith('Raises:'):
            in_params = False
        elif in_params:
            # Parse parameter line (handles "param: desc" and "- param: desc" formats)
            if ':' in stripped and (stripped[0].isalpha() or stripped.startswith(('-', '*'))):
                param_name = stripped.lstrip('- *').split(':')[0].strip()
                param_desc = ':'.join(stripped.lstrip('- *').split(':')[1:]).strip()
                params[param_name] = param_desc
                current_param = param_name
            elif current_param and stripped:
                # Continuation of previous parameter description
                params[current_param] += ' ' + stripped
    
    return params

def parse_function_definition(function_def: str) -> Dict[str, Any]:
    """Parse a function definition string to extract metadata including type hints."""
    result = {
        "name": "",
        "description": "",
        "parameters": {"type": "object", "properties": {}},
        "required": [],
        "returns": {"type": "string", "description": ""}
    }
    
    # Parse the function using AST
    tree = ast.parse(function_def.strip())
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return result
    
    func = tree.body[0]
    result["name"] = func.name
    
    # Extract docstring
    docstring = ast.get_docstring(func) or ""
    if docstring:
        # Extract description (first line/paragraph)
        desc_end = docstring.find('\n\n') if '\n\n' in docstring else docstring.find('\nArgs:')
        desc_end = desc_end if desc_end > 0 else docstring.find('\nParameters:')
        result["description"] = docstring[:desc_end].strip() if desc_end > 0 else docstring.strip()
        
        # Parse parameter descriptions
        param_descs = parse_docstring_params(docstring)
        
        # Extract return description
        if "Returns:" in docstring:
            result["returns"]["description"] = docstring.split("Returns:")[1].strip().split('\n')[0]
    
    # Extract parameters with type hints
    args = func.args
    defaults = args.defaults
    num_args = len(args.args)
    num_defaults = len(defaults)
    
    for i, arg in enumerate(args.args):
        if arg.arg == 'self':
            continue
        
        param_info = {
            "type": get_type_from_annotation(arg.annotation) if arg.annotation else "string",
            "description": param_descs.get(arg.arg, "")
        }
        
        # Check for default value
        default_idx = i - (num_args - num_defaults)
        if default_idx >= 0:
            param_info["default"] = ast.literal_eval(ast.unparse(defaults[default_idx]))
        else:
            result["required"].append(arg.arg)
        
        result["parameters"]["properties"][arg.arg] = param_info
    
    # Extract return type
    if func.returns:
        result["returns"]["type"] = get_type_from_annotation(func.returns)
    
    return result


def get_tool_descriptions(function_list):
    """Extract tool descriptions from the function list"""
    descriptions = []
    
    for function in function_list:
        function_string = inspect.getsource(function)
        result = parse_function_definition(function_string)
        
        if result:
            descriptions.append(result)
    
    return descriptions if descriptions else "Could not extract tool descriptions"


# This function is used to create chat completions with instructor and litellm


def chat_completions(
    conversation: list, 
    prompt_path: str,
    response_model: Type[Any],
    prompt_params: Optional[Dict[str, Any]] = {},
    agent_tools: Optional[List[Dict[str, Any]]] = [],
    models: List[str] = ["gpt-4.1", "groq/llama-3.3-70b-versatile"] 
) -> tuple[Any, Any]:
    """
    Create chat completions with instructor and litellm
    
    Args:
        model: List of model names for litellm router
        prompt_path: Path to the prompt template
        response_model: Pydantic model for structured output
        messages: List of chat messages
        response_model: Pydantic model for structured output
        
    Returns:
        Tuple of (parsed_response, raw_response)
    """
    # Create instructor client from litellm
    client = instructor.from_litellm(completion)
    
    response = None
    raw_response = None
    last_error = None
    
    for model in models:
        try:
            # Load model-specific prompt from YAML (uses model name as key)
            prompt_template = prompt_template_config(prompt_path, model, model=model)
            # Render the template to get the final prompt string
            prompt = prompt_template.render(
                available_tools=agent_tools,
                **prompt_params
            )
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                messages=[{"role": "system", "content": prompt}, *conversation],
                response_model=response_model,
                temperature=0
            )
            break  # Success, exit loop
        except Exception as e:
            print(f"Error with model {model}: {e}")
            last_error = e
            continue
    
    # If all models failed, raise error
    if response is None or raw_response is None:
        raise RuntimeError(f"All models failed. Last error: {last_error}")
    
    return response, raw_response
