from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolCall
from operator import add
import ast
import inspect
from api.api.agent.models import AgentResponse, ToolCall

def format_ai_message(response: AgentResponse):
    """Format the AI response into a LangChain AIMessage object"""
    if response.tool_calls:
        tool_calls = []
        for i, tc in enumerate(response.tool_calls):
            tool_calls.append({
                "id": f"call_{i}",
                "name": tc.name,
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