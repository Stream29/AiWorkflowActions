"""
Node type specific prompts for AI generation.
Provides contextual guidance for each node type to improve generation quality.
"""

from typing import Dict, Optional

from dsl_model.enums import NodeType

# Mapping of node types to their specific generation prompts
NODE_TYPE_PROMPTS: Dict[NodeType, str] = {
    NodeType.START: """
Start nodes initialize the workflow with input parameters.
Consider what initial data or parameters the workflow needs to begin processing.
Define clear parameter names and descriptions for user inputs.""",

    NodeType.END: """
End nodes define the final output of the workflow.
Specify what data should be returned to the user or calling system.
Consider aggregating or formatting results from previous nodes.""",

    NodeType.ANSWER: """
Answer nodes provide responses to users in conversational workflows.
Format the response text clearly using available variables from previous nodes.
Consider including context, formatting, and appropriate tone for user interaction.""",

    NodeType.LLM: """
LLM nodes process data using large language models.
Define clear system prompts that specify the task and expected output format.
Use variables from previous nodes to provide context in the user prompt.
Consider temperature, max tokens, and whether to enable memory for conversation continuity.""",

    NodeType.CODE: """
Code nodes execute Python code for data processing and transformation.
Write clean, efficient Python code that processes input variables and returns output.
Use descriptive variable names and include error handling where appropriate.
Remember that the code runs in a sandboxed environment with limited libraries.""",

    NodeType.HTTP_REQUEST: """
HTTP request nodes make external API calls.
Configure the method (GET, POST, etc.), URL, headers, and body as needed.
Use variables from previous nodes for dynamic URLs or request parameters.
Consider authentication requirements and error handling for failed requests.""",

    NodeType.TOOL: """
Tool nodes integrate with external tools and services.
Select the appropriate tool provider and configure required parameters.
Map input variables from previous nodes to tool parameters.
Consider the tool's output format and how it will be used in subsequent nodes.""",

    NodeType.IF_ELSE: """
If-else nodes create conditional branches in the workflow.
Define clear conditions using variables from previous nodes.
Consider all possible branches and ensure each has appropriate handling.
Use logical operators (AND, OR) for complex conditions when needed.""",

    NodeType.TEMPLATE_TRANSFORM: """
Template transform nodes format and transform text using templates.
Use Jinja2 syntax to create dynamic templates with variables.
Consider output formatting, line breaks, and special characters.
Leverage template filters for data transformation (e.g., upper, lower, date formatting).""",

    NodeType.VARIABLE_ASSIGNER: """
Variable assigner nodes create or modify workflow variables.
Assign values from previous nodes or create new computed values.
Use clear, descriptive variable names that indicate their purpose.
Consider variable types and ensure compatibility with downstream nodes.""",

    NodeType.LEGACY_VARIABLE_AGGREGATOR: """
Legacy variable aggregator nodes (variable-assigner) combine variables.
This is a legacy node type, consider using variable-aggregator instead.
Assign values from previous nodes or create new computed values.
Ensure compatibility with downstream nodes.""",

    NodeType.VARIABLE_AGGREGATOR: """
Variable aggregator nodes combine multiple variables into structured output.
Select variables from previous nodes to aggregate.
Define the output structure (object, array, or formatted text).
Consider naming conventions and data organization for clarity.""",

    NodeType.KNOWLEDGE_RETRIEVAL: """
Knowledge retrieval nodes query knowledge bases for relevant information.
Configure the query with appropriate keywords or semantic search parameters.
Consider the retrieval strategy (keyword, semantic, hybrid) based on use case.
Set appropriate top_k values and similarity thresholds.""",

    NodeType.ITERATION: """
Iteration nodes process collections of data in loops.
Define the input array or list to iterate over from previous nodes.
Configure the processing logic for each item in the iteration.
Consider output aggregation and how to handle iteration results.""",

    NodeType.ITERATION_START: """
Iteration start nodes begin a loop structure in the workflow.
Define the collection or array to iterate over.
Set up iteration variables that will be available within the loop.
Consider loop termination conditions if needed.""",

    NodeType.PARAMETER_EXTRACTOR: """
Parameter extractor nodes extract structured data from unstructured input.
Define clear extraction parameters and expected output schema.
Use appropriate extraction methods (regex, NLP, structured parsing).
Consider handling missing or malformed data gracefully.""",

    NodeType.QUESTION_CLASSIFIER: """
Question classifier nodes categorize user questions or intents.
Define clear classification categories that cover expected inputs.
Configure confidence thresholds for classification decisions.
Consider handling ambiguous or out-of-scope questions.""",

    NodeType.AGENT: """
Agent nodes orchestrate complex multi-step AI tasks.
Configure the agent's role, capabilities, and available tools.
Define clear goals and success criteria for the agent.
Consider setting appropriate iteration limits and timeout values.""",

    NodeType.LOOP: """
Loop nodes implement iterative logic in workflows.
Configure loop conditions and iteration variables.
Define clear termination criteria to prevent infinite loops.
Consider aggregating results from loop iterations.""",

    NodeType.LOOP_START: """
Loop start nodes initiate conditional loops in the workflow.
Define clear loop conditions and termination criteria.
Initialize loop variables and counters as needed.
Consider maximum iteration limits to prevent infinite loops.""",

    NodeType.LOOP_END: """
Loop end nodes complete a loop structure and define continuation logic.
Specify conditions for loop continuation or termination.
Handle loop results and aggregation of data from iterations.
Consider what data should be passed to subsequent nodes after loop completion.""",

    NodeType.DOCUMENT_EXTRACTOR: """
Document extractor nodes parse and extract data from documents.
Configure extraction rules for different document types (PDF, Word, etc.).
Define specific fields or patterns to extract.
Consider handling various document formats and encoding issues.""",

    NodeType.LIST_OPERATOR: """
List operator nodes perform operations on arrays and lists.
Configure operations like filter, map, reduce, sort, or join.
Define operation parameters and conditions clearly.
Consider handling empty lists and edge cases.""",

    NodeType.NOTE: """
Note nodes provide documentation and comments within the workflow.
Use clear, descriptive text to explain workflow logic or important details.
These nodes don't affect execution but improve workflow maintainability.
Consider adding notes at complex decision points or integration points.""",
}


def get_node_type_prompt(node_type: NodeType) -> str:
    """
    Get the specific prompt for a node type.

    Args:
        node_type: The type of node to get prompt for

    Returns:
        Node-specific prompt string, or empty string if type not found
    """
    return NODE_TYPE_PROMPTS.get(node_type, "")


def get_node_type_prompt_by_string(node_type_str: str) -> str:
    """
    Get the specific prompt for a node type by string value.
    This is a convenience method for when you have a string representation.

    Args:
        node_type_str: The string value of the node type

    Returns:
        Node-specific prompt string, or empty string if type not found
    """
    # Try to find the matching NodeType enum
    for node_type in NodeType:
        if node_type.value == node_type_str:
            return get_node_type_prompt(node_type)
    return ""