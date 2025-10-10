"""
Node type specific prompts for AI generation.
Provides contextual guidance for each node type to improve generation quality.
"""

from typing import Dict, Optional

from dsl_model.enums import NodeType

# Mapping of node types to their specific generation prompts
NODE_TYPE_PROMPTS: Dict[NodeType, str] = {
    NodeType.START: """
Start nodes initialize the workflow with input parameters and system variables.
- Outputs all user-provided input variables (custom keys defined by user)
- Outputs 15 system variables: sys.query, sys.files, sys.conversation_id, sys.user_id, sys.app_id, sys.workflow_id, sys.workflow_run_id, sys.dialogue_count, sys.time (ISO 8601 UTC), sys.timezone (e.g., America/New_York), sys.timestamp (Unix timestamp), sys.random_id (8-char random), sys.metadata (dict), sys.settings (dict), sys.current_date (YYYY-MM-DD)
- Consider what initial data your workflow needs to function properly
- Define clear, semantic parameter names that describe the purpose of each input""",

    NodeType.END: """
End nodes define the final output of the workflow by collecting values from previous nodes.
- Outputs dynamic variables based on configured VariableSelectors (outputs field)
- Each output maps a variable name to a value_selector path: {"variable": "result", "value_selector": ["llm_node", "text"]}
- Typically used to aggregate and return final results, summaries, or processed data
- Consider what information the workflow caller or user needs to receive
- Use semantic variable names that clearly indicate what each output represents""",

    NodeType.ANSWER: """
Answer nodes render and deliver responses to users in conversational workflows.
- Outputs: answer (str, rendered Markdown text), files (array, extracted files from template)
- Supports variable interpolation using {{#node_id.variable#}} syntax in answer templates
- Automatically extracts FileSegments and ArrayFileSegments from rendered content
- Use Markdown formatting for rich text responses (headers, lists, code blocks, links)
- Consider tone, clarity, and user context when crafting response templates
- Include relevant data from previous nodes to provide informative answers""",

    NodeType.LLM: """
LLM nodes invoke large language models for text generation and reasoning tasks.
- Outputs: text (str), reasoning_content (str, extracted from <think> tags if reasoning_format=separated), usage (dict, token stats), finish_reason (str|null), structured_output (optional, enabled via config), files (optional, multimodal outputs)
- reasoning_format: "tagged" (keeps <think> tags in text) vs "separated" (extracts reasoning to reasoning_content)
- Supports conversation memory, vision inputs, and structured output schemas
- Write clear system prompts that define the task, role, and expected output format
- Use variables from previous nodes in user prompts to provide necessary context
- Consider temperature (creativity), max_tokens (length limit), and top_p for generation control
- Enable memory for multi-turn conversations requiring context continuity""",

    NodeType.CODE: """
Code nodes execute Python or JavaScript code in a sandboxed environment.
- Outputs: <custom> (dynamic, defined by code's return dict keys)
- Supported output types: string, number, boolean, object, array[string], array[number], array[object], array[boolean]
- Automatic type validation against output schema if defined
- Boolean-to-integer conversion: True→1, False→0 when schema expects number
- Write clean, efficient code that processes input variables and returns a dictionary
- Use descriptive variable names and handle edge cases gracefully
- Remember: limited library access, max string length 400K chars, max nesting depth 5, array length limits apply
- Define meaningful output schema to enable validation and type safety""",

    NodeType.HTTP_REQUEST: """
HTTP request nodes make external API calls with intelligent file detection.
- Outputs: status_code (int), body (str, empty if files detected), headers (dict), files (array)
- Automatically detects files via Content-Disposition header, MIME type analysis, and content inspection
- File detection prioritizes: Content-Disposition > Content-Type classification > binary content analysis
- Configure HTTP method (GET, POST, PUT, PATCH, DELETE), URL, headers (auth, content-type), and body
- Use variables from previous nodes for dynamic URLs, query parameters, or request payloads
- Support JSON, form-data, x-www-form-urlencoded body types
- Consider authentication (Bearer tokens, API keys, Basic Auth) and error handling strategies
- body field becomes empty string when files are detected to avoid duplication""",

    NodeType.TOOL: """
Tool nodes invoke external tools (built-in, API, or plugins) with rich output types.
- Outputs: text (str, from TEXT/LINK messages), files (array, from IMAGE/FILE messages), json (list[dict], defaults to [{"data": []}]), <tool_variables> (optional custom outputs via VARIABLE messages)
- Processes stream-based ToolInvokeMessages: TEXT, JSON, FILE, IMAGE_LINK, BINARY_LINK, VARIABLE, LOG
- json field always exists and is always a list, providing default [{"data": []}] if no JSON messages
- Tool variables allow tools to define custom outputs beyond standard text/files/json
- Select appropriate tool provider (e.g., google_search, dalle, serpapi) and configure required parameters
- Map workflow variables to tool input parameters accurately
- Consider tool output structure when designing downstream nodes
- Reserved output names: json, text, files cannot be used for tool_variables""",

    NodeType.IF_ELSE: """
If-else nodes create conditional branches based on logical evaluations.
- Outputs: result (bool, final condition result), selected_case_id (str, ALWAYS string never None)
- selected_case_id: "case_abc123" (matched case), "false" (else branch), or "true"/"false" (legacy)
- selected_case_id determines which downstream edge to follow via edge_source_handle
- Supports new cases structure (multiple conditions with IDs) and legacy conditions structure
- Define clear, testable conditions using comparison operators (=, ≠, <, >, ≤, ≥, contains, is empty, is not empty)
- Support logical operators (AND, OR, NOT) for complex multi-condition evaluations
- Consider all possible branches and ensure proper routing for each case
- Use semantic case names that clearly indicate what condition is being tested""",

    NodeType.TEMPLATE_TRANSFORM: """
Template transform nodes render Jinja2 templates with workflow variables.
- Outputs: output (str, rendered template result, max 400K characters)
- Full Jinja2 syntax support: variable interpolation, loops, conditionals, filters, macros
- Variables retrieved from VariablePool and converted to Python objects via .to_object()
- Execution via CodeExecutor with CodeLanguage.JINJA2
- Use Jinja2 filters for data transformation: upper, lower, title, length, join, replace, date formatting
- Support complex templates with loops ({% for %}), conditionals ({% if %}), and nested structures
- Consider output formatting, line breaks, HTML/XML escaping, and special character handling
- Ideal for generating formatted text, HTML reports, JSON strings, or configuration files""",

    NodeType.VARIABLE_ASSIGNER: """
Variable assigner nodes modify VariablePool variables in-place (v2, NO output variables).
- Outputs: NONE (empty dict, modifies existing variables directly in VariablePool)
- Supports operations: SET, OVER_WRITE, CLEAR, ADD, SUBTRACT, MULTIPLY, DIVIDE, APPEND, EXTEND, REMOVE_FIRST, REMOVE_LAST
- Modifies variables via variable_pool.add(selector, updated_value), downstream nodes see changes
- Persists conversation variables to database when variable selector starts with CONVERSATION_VARIABLE_NODE_ID
- Use for: in-place variable updates, accumulation, counter increments, list manipulation, state management
- Consider operation semantics: SET (initialize), OVER_WRITE (force replace), arithmetic ops (ADD/SUBTRACT/etc.), list ops (APPEND/EXTEND/REMOVE)
- Ensure variable exists before using non-SET operations to avoid errors
- Choose meaningful variable names that indicate purpose and scope""",

    NodeType.LEGACY_VARIABLE_AGGREGATOR: """
Legacy variable aggregator nodes (variable-assigner) combine variables.
This is a legacy node type, consider using variable-aggregator instead.
Assign values from previous nodes or create new computed values.
Ensure compatibility with downstream nodes.""",

    NodeType.VARIABLE_AGGREGATOR: """
Variable aggregator nodes select the first non-None value from multiple variables.
- Outputs (simple mode): output (Segment, first non-None variable as Segment object)
- Outputs (grouped mode): <group_name> (dict with {"output": Segment} for each group)
- Returns Segment objects, not .value - downstream nodes auto-resolve Segments
- Simple mode: iterates variables list, returns first non-None as {"output": variable}
- Grouped mode: each group gets first non-None from its variables as {group_name: {"output": variable}}
- Use for: fallback chains, default value handling, multi-source data aggregation, conditional selection
- If all variables are None, outputs empty dict (simple mode) or group omitted (grouped mode)
- Consider variable ordering carefully as first non-None wins
- Useful for implementing graceful degradation when primary data sources fail""",

    NodeType.KNOWLEDGE_RETRIEVAL: """
Knowledge retrieval nodes search knowledge bases with semantic or keyword matching.
- Outputs: result (ArrayObjectSegment, array of document chunks with metadata)
- Each result object contains: content (str, document text or Q&A format), title (str), metadata (dict with 17+ fields)
- Metadata fields: _source, dataset_id, dataset_name, document_id, document_name, data_source_type, segment_id, retriever_from, score (float 0-1), position (int, 1-based rank), child_chunks (list), segment_hit_count, segment_word_count, segment_position, segment_index_node_hash, doc_metadata (custom fields)
- External KB: data_source_type="external", metadata includes raw doc_metadata from external API
- Results sorted by score descending, then assigned position (1, 2, 3...)
- Configure query dynamically using variables, select retrieval mode (semantic, keyword, hybrid)
- Set top_k (result count limit) and score_threshold (minimum similarity)
- Consider reranking strategy for improved result quality""",

    NodeType.ITERATION: """
Iteration nodes execute sub-workflows on array elements with AUTOMATIC flattening.
- Outputs: output (list[object] or ArraySegment, automatically flattened if ALL non-None outputs are lists)
- Flattening is AUTOMATIC, not configurable: checks if all non-None outputs are lists, then uses extend()
- Non-list outputs or mixed outputs → nested structure preserved
- Empty input (NoneSegment or empty array) → empty output array
- None outputs from failed iterations: included in nested mode, excluded in flattened mode
- Supports error handling: continued (skip failures), terminated (stop on first failure)
- Define input array variable and configure sub-workflow logic for each element
- Consider max parallelism for performance vs resource usage
- Ideal for: batch processing, data transformation pipelines, parallel API calls
- IMPORTANT: Flattening behavior is determined by output data structure, not by configuration""",

    NodeType.ITERATION_START: """
Iteration start nodes begin a loop structure in the workflow.
Define the collection or array to iterate over.
Set up iteration variables that will be available within the loop.
Consider loop termination conditions if needed.""",

    NodeType.PARAMETER_EXTRACTOR: """
Parameter extractor nodes use LLM to extract structured parameters from text.
- Outputs: __is_success (int, 1=success 0=failure), __reason (str, failure reason), __usage (dict, token stats), <param_name> (dynamic, extracted parameters per schema)
- Uses LLM with structured output or schema-guided generation to extract parameters
- Parameter schema defines expected fields, types, descriptions, and requirements (required vs optional)
- Supports complex types: string, number, boolean, array, object, nested structures
- Define clear parameter schemas with meaningful names and comprehensive descriptions
- Provide extraction instructions that guide the LLM on what to look for and how to interpret ambiguous cases
- Consider validation rules, default values, and error handling for missing/malformed data
- Use query variable to specify which text content to extract parameters from
- Ideal for: form filling, entity extraction, structured data mining from natural language""",

    NodeType.QUESTION_CLASSIFIER: """
Question classifier nodes use LLM to categorize user questions into predefined classes.
- Outputs: class_name (str, matched category name), class_id (str, matched category ID used for routing), usage (dict, token stats)
- class_id serves as edge_source_handle for branch routing to downstream nodes
- Uses few-shot prompting with chat models or completion models
- Supports vision inputs for multimodal classification
- Define clear, non-overlapping classification categories with descriptive names
- Provide classification instructions to guide the LLM's decision-making process
- Configure classes with unique IDs that map to downstream workflow branches
- Support conversation memory for context-aware classification
- Consider edge cases: ambiguous questions, out-of-scope inputs, multi-intent queries
- Ideal for: intent routing, topic categorization, customer support triage""",

    NodeType.AGENT: """
Agent nodes execute autonomous reasoning loops with tool calling capabilities.
- Outputs: text (str, final response), usage (dict, total token stats), files (array), json (list[dict], ALWAYS list combining agent logs + JSON messages, defaults to [{"data": []}]), <tool_variables> (optional custom outputs)
- json field structure: [agent_log_dict1, agent_log_dict2, ..., json_message1, json_message2, ...] or [{"data": []}] if empty
- Agent logs capture reasoning process: {id, parent_id, status, data, label, metadata, node_id}
- Supports Plugin Agent Strategy with streaming, multi-turn conversations, and tool orchestration
- Configure agent prompt (role, instructions), max iterations, tool selection strategy
- Define available tools the agent can invoke during reasoning process
- Support conversation memory for context continuity across turns
- Consider iteration limits to prevent infinite loops and control costs
- Ideal for: complex multi-step tasks, research agents, autonomous problem-solving, tool orchestration""",

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
Document extractor nodes extract text content from 40+ file formats.
- Outputs (single file): text (str, extracted text content)
- Outputs (file array): text (ArrayStringSegment, one string per file)
- Supports documents: .txt, .md, .html, .pdf, .doc, .docx, .xls, .xlsx, .csv, .ppt, .pptx, .epub, .eml, .msg
- Supports data formats: .json, .yaml, .yml, .properties
- Supports code files: .py, .js, .ts, .java, .cpp, .go, .rs, .rb, .php, .swift, .kt, .scala, .sh, .sql, .css, etc.
- Supports subtitles: .vtt
- Special handling: tables → Markdown format, character encoding auto-detection (chardet), external API support for complex formats
- Configure input variable pointing to FileSegment or ArrayFileSegment
- Consider file size limits, encoding issues, and format-specific extraction quality
- Ideal for: document preprocessing, content analysis, data extraction from uploads""",

    NodeType.LIST_OPERATOR: """
List operator nodes perform filter, extract, order, and limit operations on arrays.
- Outputs: result (array, same type as input), first_record (element or None), last_record (element or None)
- Supports array types: ArrayStringSegment, ArrayNumberSegment, ArrayFileSegment, ArrayBooleanSegment
- Operations applied in order: 1) Filter (conditions), 2) Extract (by serial number), 3) Order (asc/desc), 4) Limit (first N)
- Filter: comparison operators (=, ≠, <, >, ≤, ≥, contains, is empty, is not empty), logical operators (AND, OR)
- Extract: 1-based indexing, result is single-element array
- Order: sort by value or file properties (name, size, type, extension)
- Limit: take first N elements after filtering/sorting
- Empty input → result=[], first_record=None, last_record=None
- Ideal for: data filtering, top-K selection, array manipulation, file list processing""",

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