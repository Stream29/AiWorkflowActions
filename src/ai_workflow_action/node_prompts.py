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
- Outputs system variables depending on workflow context:
  - Core variables (always available): sys.files (array, uploaded files), sys.user_id (str), sys.app_id (str), sys.workflow_id (str), sys.workflow_run_id (str, workflow execution ID)
  - Chat workflow variables: sys.query (str, user query), sys.conversation_id (str), sys.dialogue_count (int, message count in conversation)
  - RAG pipeline variables: sys.document_id (str), sys.original_document_id (str), sys.batch (str), sys.dataset_id (str), sys.datasource_type (str), sys.datasource_info (dict), sys.invoke_from (str)
- System variables are automatically populated by the workflow runtime and cannot be manually set
- Some system variables may be None depending on workflow type (e.g., sys.query and sys.conversation_id only exist in chat workflows)
- Consider what initial data your workflow needs to function properly
- Define clear, semantic parameter names that describe the purpose of each input
- Access system variables in downstream nodes using selectors like {{#sys.query#}} or {{#sys.files#}}""",

    NodeType.END: """
End nodes define the final output of the workflow by collecting values from previous nodes.
- Outputs dynamic variables based on configured VariableSelectors (outputs field)
- Each output maps a variable name to a value_selector path: {"variable": "result", "value_selector": ["llm_node", "text"]}
- Typically used to aggregate and return final results, summaries, or processed data
- Consider what information the workflow caller or user needs to receive
- Use semantic variable names that clearly indicate what each output represents""",

    NodeType.ANSWER: """
Answer nodes render and deliver responses to users in conversational workflows.
- Outputs: answer (str, rendered via SegmentGroup.markdown property), files (ArrayFileSegment, extracted files from template variables)
- Template structure: answer field contains plain text + variable placeholders (e.g., "prefix{{#sys.query#}}suffix")
- Variable syntax: {{#node_id.variable#}} where node_id can be node ID, sys, or special node identifiers
- Template processing: VariablePool.convert_template() splits template by VARIABLE_PATTERN regex, resolves variables, returns SegmentGroup
- Output generation: segments.markdown concatenates each segment's markdown property (strings remain strings, arrays become bullet lists, files become markdown links/images, objects become JSON)
- File extraction: _extract_files_from_segments() iterates segments, extracts File objects from FileSegment (single file) and ArrayFileSegment (multiple files), flattens into single list
- FileSegment markdown format: Images as ![filename](url), other files as [filename](url)
- ArrayFileSegment markdown format: Each file on separate line, newline-joined
- Use variable references to dynamically include LLM outputs, system variables, or file outputs from previous nodes
- Craft templates that combine static text with variable interpolation for context-aware responses
- Consider Markdown formatting capabilities: headers (# ## ###), lists (- item), code blocks (```lang```), links ([text](url)), emphasis (*italic* **bold**)
- Example templates: "Here is your result: {{#llm.text#}}", "Query: {{#sys.query#}}\n\nAnswer: {{#llm.text#}}\n\nFiles: {{#http_request.files#}}"
- Template validation: variables must match pattern [a-zA-Z0-9_]{1,50}(.[a-zA-Z_][a-zA-Z0-9_]{0,29}){1,10}, enclosed in {{##}}
- Empty or None variable values render as empty strings in final output""",

    NodeType.LLM: """
LLM nodes invoke large language models for text generation and reasoning tasks.
- Outputs: text (str), reasoning_content (str, always present), usage (dict, token stats), finish_reason (str|null), structured_output (optional, enabled via structured_output_switch_on), files (optional, multimodal image/video/audio/document outputs via vision-capable models)
- reasoning_format controls <think> tag handling: "tagged" (default, keeps tags in text, reasoning_content=""), "separated" (removes tags from text, extracts content to reasoning_content for workflow access via {{#node_id.reasoning_content#}})
- Supports conversation memory (multi-turn context via memory config), vision inputs (images/videos/documents via vision.enabled), and structured output schemas (JSON schema validation)
- structured_output requires structured_output_switch_on=true and valid JSON schema in structured_output.schema field
- Write clear system prompts that define the task, role, and expected output format
- Use variables from previous nodes in user prompts to provide necessary context
- Consider temperature (creativity 0-1), max_tokens (length limit), top_p (nucleus sampling), and presence_penalty/frequency_penalty for generation control
- Enable memory for multi-turn conversations requiring context continuity, supports both chat models (role-based messages) and completion models (with role_prefix configuration)
- Vision support allows multimodal inputs (sys.files or custom file variables) and outputs (files array contains generated images/media from model plugins)""",

    NodeType.CODE: """
Code nodes execute Python or JavaScript code in a sandboxed remote execution environment.
- Outputs: <custom> (dynamic, defined by code's return dict keys)
- Supported output types: string, number, boolean, object, array[string], array[number], array[object], array[boolean]
- Automatic type validation against output schema if defined
- Boolean-to-integer conversion: True→1, False→0 when schema expects number or array[number]
- Write clean, efficient code that processes input variables and returns a dictionary
- Use descriptive variable names and handle edge cases gracefully
- Constraints: max string length 400K chars, max nesting depth 5, array[string] max 30 elements, array[number] max 1000 elements, array[object] max 30 elements, array[boolean] no specific limit, number range -9223372036854775808 to 9223372036854775807, float precision max 20 decimal digits
- Execution environment: Python 3 or JavaScript/Node.js in remote sandbox with network access enabled, basic standard library imports available
- Define meaningful output schema to enable validation and type safety""",

    NodeType.HTTP_REQUEST: """
HTTP request nodes make external API calls with intelligent file detection.
- Outputs: status_code (int), body (str, empty if files.value non-empty), headers (dict), files (ArrayFileSegment)
- HTTP methods: GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS (case-insensitive)
- Body types: none, json, form-data, x-www-form-urlencoded, raw-text, binary
- File detection logic: 1) Content-Disposition header (attachment/filename), 2) MIME type classification (excludes text/*, json, xml, yaml, graphql unless csv), 3) UTF-8 decode attempt with text marker detection (excludes {, [, <, function, var, const, let)
- body field is response.text if files.value is empty, else empty string to avoid duplication
- Authentication types: no-auth, api-key with config.type (bearer, basic, custom), custom header support via config.header
- Bearer auth: Authorization header with "Bearer {api_key}", Basic auth: "Basic {base64(credentials)}", Custom auth: {config.header}: {api_key}
- form-data supports both FileSegment and ArrayFileSegment, allows multiple files per key
- Timeout configuration: connect, read, write timeouts (all configurable)
- SSL verification: ssl_verify flag controls certificate validation
- Retry support: retry_config with retry_enabled, max_retries, retry_interval
- Use variables from previous nodes for dynamic URLs, query parameters, headers, body content
- Variable template parsing: URL, headers, params, body fields all support {{#node.variable#}} syntax
- Content-Type auto-handling: automatically sets for json/form-data/x-www-form-urlencoded, manually removable for multipart boundary management
- Response size limits: different thresholds for text vs binary content
- Example auth configs: Bearer token for API access, Basic auth for legacy systems, custom header for proprietary APIs""",

    NodeType.TOOL: """
Tool nodes invoke external tools (built-in, API, or plugins) with rich output types.
- Outputs: text (str, from TEXT/LINK messages), files (array, from IMAGE/FILE/IMAGE_LINK/BINARY_LINK/BLOB messages), json (list[dict], empty list becomes [{"data": []}]), <tool_variables> (optional custom outputs via VARIABLE messages)
- Processes 12 ToolInvokeMessage types: TEXT, IMAGE, LINK, BLOB, JSON, IMAGE_LINK, BINARY_LINK, VARIABLE, FILE, LOG, BLOB_CHUNK, RETRIEVER_RESOURCES
- json field behavior: if JSON messages exist, extends json list with message objects; if no JSON messages, returns [{"data": []}] as fallback
- Tool variables: custom named outputs with types dict, list, str, int, float, bool (basic types and lists only)
- Variable streaming: when VariableMessage.stream=True, variable_value must be string and gets concatenated incrementally; when stream=False, variable_value set directly (any supported type)
- LINK messages: formatted as "Link: {url}\n" and appended to text output
- Select appropriate tool provider (e.g., google_search, dalle, serpapi) and configure required parameters
- Map workflow variables to tool input parameters accurately
- Consider tool output structure when designing downstream nodes
- Reserved output names: json, text, files are built-in outputs and cannot be used for tool_variables (validation raises ValueError)""",

    NodeType.IF_ELSE: """
If-else nodes create conditional branches based on logical evaluations.
- Outputs: result (bool, final condition result), selected_case_id (str, ALWAYS string never None)
- selected_case_id determines edge routing: "case_abc123" (matched case), "false" (no case matched/else branch), "true"/"false" (legacy mode)
- edge_source_handle = selected_case_id or "false" (fallback ensures routing always works)
- Structure: NEW cases (list[Case] with case_id, logical_operator, conditions) vs LEGACY (deprecated top-level conditions + logical_operator fields)
- Each Case: case_id (str, must match downstream edge.sourceHandle), logical_operator (Literal["and", "or"]), conditions (list[Condition])
- Condition: variable_selector (path to variable), comparison_operator (see full list below), value (expected value), optional sub_variable_condition (for file arrays)
- Comparison operators (21 total):
  String/Array: "contains", "not contains", "start with", "end with", "is", "is not", "empty", "not empty", "in", "not in", "all of"
  Number: "=", "≠", ">", "<", "≥", "≤", "null", "not null"
  File: "exists", "not exists"
- Logical operators: "and" (all conditions must pass), "or" (any condition passes) - NO "not" operator exists
- Case evaluation: iterates cases in order, returns first matching case's case_id, short-circuits on match
- Legacy mode: single condition group, outputs selected_case_id="true" if pass, "false" if fail
- Sub-conditions: for ArrayFileSegment with operators "contains"/"not contains"/"all of", use sub_variable_condition to check file attributes (name, type, size, extension)
- Edge routing: downstream edges must have sourceHandle matching case_id values, "false" handles else/no-match branch
- Example: Case(case_id="high_priority", logical_operator="and", conditions=[...]) requires edge with sourceHandle="high_priority"
- Consider all possible branches, ensure every case_id has corresponding edge, always provide "false" edge for else path
- Use semantic case_id values that clearly indicate condition intent (e.g., "user_verified", "amount_exceeds_limit", "file_type_pdf")""",

    NodeType.TEMPLATE_TRANSFORM: """
Template transform nodes render Jinja2 templates with workflow variables into text output.
- Outputs: output (str, rendered template result, max 400,000 characters)
- Inputs: variables (list[VariableSelector]) maps template variable names to workflow variable selectors
- Variable conversion: VariablePool.get() returns Segment, then .to_object() converts to Python primitives (str, int, float, bool, dict, list, File, or None)
- None handling: missing or None variables become None in template context, use default filter or conditionals to handle: {{ variable | default("fallback") }} or {% if variable %}
- Execution: CodeExecutor with CodeLanguage.JINJA2 in remote Python sandbox, uses standard jinja2.Template(template).render(**inputs)
- Full Jinja2 feature support: variable interpolation {{ var }}, loops {% for item in items %}, conditionals {% if condition %}, filters {{ text|upper }}, tests {% if var is defined %}, macros {% macro name() %}, template inheritance {% extends %}, includes {% include %}, whitespace control {%- -%}, comments {# comment #}
- Built-in filters: upper, lower, capitalize, title, trim, length, reverse, sort, join, replace, default, first, last, sum, abs, round, int, float, string, list, dictsort, groupby, slice, batch, tojson, safe, escape, urlize, and 50+ more from standard Jinja2
- Built-in tests: defined, undefined, none, number, string, sequence, mapping, boolean, true, false, even, odd, divisibleby, equalto, in, and 20+ more
- Math operations: +, -, *, /, //, %, ** directly in templates
- Logic operations: and, or, not, comparison operators (==, !=, <, >, <=, >=, in)
- Variable access: dot notation (obj.key) and bracket notation (obj['key']) for dicts, array indexing (arr[0]), attribute chaining (obj.nested.value)
- String formatting: use format filter {{ "%s is %d" | format(name, age) }} or f-string-like patterns with variables
- Whitespace control: {%- for item in list -%} removes whitespace before/after tags
- Error handling: template syntax errors or undefined variables (without default filter) cause CodeExecutionError with error message, node fails with WorkflowNodeExecutionStatus.FAILED
- Security: sandboxed execution, no filesystem access, no dangerous builtins (open, eval, exec unavailable)
- Use cases: formatted reports, HTML/XML generation, JSON string assembly, configuration files, email templates, multi-line text with dynamic content
- Examples:
  Simple: "Hello {{ name }}, you have {{ count }} items"
  Loop: "{% for user in users %}{{ user.name }}: {{ user.email }}\n{% endfor %}"
  Conditional: "{% if score >= 90 %}Excellent{% elif score >= 60 %}Pass{% else %}Fail{% endif %}"
  Filters: "{{ text | upper | replace('HELLO', 'HI') }}"
  Default: "{{ optional_var | default('N/A') }}"
  JSON: "{{ {'key': value, 'items': items} | tojson }}""",

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
Legacy variable aggregator nodes (type: "variable-assigner") select the first non-None value from multiple variables.
This is a legacy type identifier that maps to the same implementation as VARIABLE_AGGREGATOR.
- Outputs (simple mode): {"output": Segment} where Segment is the first non-None variable found
- Outputs (grouped mode): {group_name: {"output": Segment}} structure
- Simple mode: iterates variables list sequentially, breaks immediately on first non-None
- Grouped mode: for each group independently, selects first non-None variable
- If all variables None: returns empty dict {}
- Use for: fallback chains, default value handling, multi-source data aggregation
IMPORTANT: Despite the type name "variable-assigner", this node does NOT assign/modify variables. It only SELECTS from existing variables.
NOTE: For actual variable assignment operations (SET, ADD, etc.), use NodeType.VARIABLE_ASSIGNER (type: "assigner") instead.""",

    NodeType.VARIABLE_AGGREGATOR: """
Variable aggregator nodes select the first non-None value from multiple variables with short-circuit evaluation.
- Outputs (simple mode): {"output": Segment} where Segment is the first non-None variable found
- Outputs (grouped mode): {group_name: {"output": Segment}} structure, only groups with non-None values appear in outputs
- Returns raw Segment objects: downstream nodes receive Segment instances, variable_pool.get() automatically calls .to_object() when accessed to convert to Python primitives (str, int, float, bool, dict, list, File, None)
- Simple mode: iterates variables list sequentially, breaks immediately on first non-None, returns {"output": variable_segment}
- Grouped mode: for each group independently, iterates group's variables sequentially, breaks on first non-None, adds {group_name: {"output": variable_segment}} to outputs; groups with all None variables are completely omitted from outputs dict
- If all variables None (simple mode): returns empty dict {}
- If all groups have only None variables (grouped mode): returns empty dict {} (no group keys present)
- Variable ordering is critical: first non-None in list/group wins and stops further iteration within that scope
- Use for: fallback chains (try primary source, then secondary, then tertiary), default value handling, multi-source data aggregation, conditional variable selection based on availability
- Example grouped mode scenario: Group1 checks [llm_output, cached_result, default_value], Group2 checks [api_response, fallback_api, static_value] - each group independently selects first available non-None value
- Useful for implementing graceful degradation when primary data sources fail or return None""",

    NodeType.KNOWLEDGE_RETRIEVAL: """
Knowledge retrieval nodes search knowledge bases using semantic, full-text, hybrid, or keyword retrieval.
- Outputs: result (ArrayObjectSegment, array of document chunks with metadata)
- Each result object contains: content (str, document text or Q&A format), title (str), metadata (dict)
- Internal Dify KB metadata fields (15 total): _source (always "knowledge"), dataset_id (str), dataset_name (str), document_id (str), document_name (str), data_source_type (str), segment_id (str), retriever_from (always "workflow"), score (float, unbounded similarity score), position (int, 1-based rank assigned after sorting), child_chunks (list[dict] with id/content/position/score fields), segment_hit_count (int), segment_word_count (int), segment_position (int), segment_index_node_hash (str), doc_metadata (dict, custom document metadata)
- External KB metadata fields (8 total): _source ("knowledge"), dataset_id, dataset_name, document_id (or title fallback), document_name (from title), data_source_type ("external"), retriever_from ("workflow"), score (float from external API), doc_metadata (dict, raw metadata from external API)
- Child chunks structure: each item contains id (str), content (str), position (int), score (float) - only present for parent-child index documents
- Results sorted by score descending (None scores treated as 0.0), then assigned position (1, 2, 3...)
- Content format: Q&A documents return "question:{question} \nanswer:{answer}", regular documents return segment content
- Retrieval mode configuration: "single" (agent-based dataset selection with LLM) or "multiple" (search all specified datasets)
- Multiple mode options: top_k (result count), score_threshold (minimum score filter), reranking_mode ("reranking_model" with reranking model config, or "weighted_score" with vector/keyword weights), reranking_enable (bool)
- Search methods (4 types): semantic_search (vector similarity), full_text_search (BM25), hybrid_search (combines semantic + full-text), keyword_search (keyword matching)
- Configure query dynamically using query_variable_selector pointing to a string variable
- Metadata filtering: supports disabled/automatic/manual modes with comparison operators (contains, not contains, start with, end with, is, is not, empty, not empty, in, not in, =, ≠, >, <, ≥, ≤, before, after) and logical operators (and, or)
- Example usage: Query from {{#start.sys.query#}}, retrieval_mode="multiple", top_k=5, score_threshold=0.3, reranking enabled for quality improvement""",

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
Iteration start nodes mark the entry point of an iteration sub-graph execution.
- Outputs: NONE (no execution logic, immediately succeeds)
- Execution behavior: returns NodeRunResult(status=SUCCEEDED) with no processing
- Purpose: serves as structural marker for iteration sub-graph entry point, referenced by parent iteration node's start_node_id field
- Graph structure: automatically created as child of iteration node with id={iteration_id}start, parentId={iteration_id}, all child nodes within iteration have iteration_id field matching parent iteration node ID
- Events: filtered during event streaming (iteration_start events not yielded to parent at line 535: if event.node_type == NodeType.ITERATION_START: continue)
- Node data: minimal IterationStartNodeData with only base fields (title, desc, error_strategy, retry_config, default_value_dict), no custom configuration
- Iteration variables: automatically injected by parent iteration node into variable pool as [iteration_id, "item"] (current element) and [iteration_id, "index"] (0-based position), NOT configured on iteration-start node
- Collection definition: handled by parent iteration node's iterator_selector field, NOT by iteration-start node
- Use cases: required structural element for all iteration nodes, automatically created by workflow editor when adding iteration node
- Not configurable: no custom logic or parameters, purely structural placeholder for iteration sub-graph entry point
- Differs from LOOP_START: iteration-start is for array-based iteration nodes (process each element), loop-start is for counter-based loop nodes (fixed iterations with break conditions)
- Consider: placement as first node in iteration sub-graph, ensure iteration_id field on all child nodes matches parent iteration node ID, never manually configure this node as it's auto-managed by parent iteration node""",

    NodeType.PARAMETER_EXTRACTOR: """
Parameter extractor nodes use LLM to extract structured parameters from text with two reasoning modes.
- Outputs: __is_success (int, 1=success 0=failure), __reason (str|None, error/validation message or None on success), __usage (dict, LLM token stats with total_tokens/total_price/currency), <param_name> (dynamic, extracted parameters per schema definition)
- Reasoning modes: "function_call" (default, uses tool calling with extract_parameters function if model supports TOOL_CALL/MULTI_TOOL_CALL features) or "prompt" (prompt engineering with few-shot examples for chat models or completion models)
- Parameter schema (get_parameter_json_schema): JSON Schema format with type, properties, required fields, optional enum for string options
- Supported parameter types: string, number, boolean, array[string], array[number], array[object], array[boolean] (legacy "bool" and "select" types auto-convert to boolean and string)
- Type validation: validates extracted parameters against schema, checks required fields, validates types with SegmentType.is_valid(), enforces enum constraints for select-type parameters
- Type transformation: numbers accept int/float/bool/string with coercion, booleans accept bool/int, arrays transform nested elements by type, missing parameters get defaults (empty array, empty string, 0, false)
- Query variable (query): VariableSelector pointing to text content to extract from, converted to string via variable.text
- Instruction field (instruction): optional custom guidance for LLM, supports variable templates ({{#node.var#}} syntax), included in system prompts
- Vision support: vision.enabled flag with vision.configs.variable_selector for file inputs, vision.configs.detail for image detail level (high/low/auto)
- Memory support: optional conversation memory via memory.window.size for multi-turn context-aware extraction
- Function call mode: generates system prompt with FUNCTION_CALLING_EXTRACTOR_SYSTEM_PROMPT (includes histories and instruction), adds few-shot examples (FUNCTION_CALLING_EXTRACTOR_EXAMPLE), creates PromptMessageTool with extract_parameters function and schema
- Prompt engineering mode: chat models use CHAT_GENERATE_JSON_PROMPT system message with few-shot examples (CHAT_EXAMPLE), completion models use COMPLETION_GENERATE_JSON_PROMPT template
- Result extraction: parses tool call arguments (function_call mode) or extracts JSON from text response (prompt mode), validates and transforms all parameters
- Reserved parameter names: __reason, __is_success are forbidden in parameter schemas (validation error)
- Failure handling: returns __is_success=0 with __reason on model invocation errors, validation failures, or JSON extraction failures
- Default fallback: if extraction completely fails, generates empty defaults (number=0, boolean=false, string="")
- Define parameter schemas with clear descriptions to guide extraction accuracy
- Use instruction field for domain-specific rules, ambiguity resolution, or format preferences
- Example schema: {"name": {"type": "string", "description": "Full name of person"}, "age": {"type": "number", "description": "Age in years", "required": true}, "interests": {"type": "array", "items": {"type": "string"}, "description": "List of hobbies"}}
- Ideal for: structured form filling, entity extraction from documents, multi-field data mining, conversation parameter collection, invoice/receipt parsing""",

    NodeType.QUESTION_CLASSIFIER: """
Question classifier nodes use LLM to categorize user questions into predefined classes.
- Outputs: class_name (str, matched category name), class_id (str, matched category ID used for routing), usage (dict, token stats)
- class_id serves as edge_source_handle for branch routing to downstream nodes
- Classes configuration: list of objects with id (str) and name (str) fields, e.g., [{"id": "f5660049-284f-41a7-b301-fd24176a711c", "name": "Customer Service"}, {"id": "8d007d06-f2c9-4be5-8ff6-cd4381c13c60", "name": "Satisfaction"}]
- LLM output format: expects JSON {"keywords": [...], "category_id": "uuid", "category_name": "..."}, keywords extracted but not returned
- Fallback behavior: if LLM output parsing fails or category_id not found in classes, defaults to first class in list (classes[0])
- Uses few-shot prompting: 2 user/assistant examples for chat models (customer service feedback, restaurant experience), embedded examples for completion models
- Instruction field: optional classification guidance text supporting variable references via {{#node.variable#}} syntax
- Supports vision inputs for multimodal classification via vision.enabled and vision.configs.variable_selector
- Supports conversation memory for context-aware classification via memory config with window.size
- Edge routing: downstream edges must have sourceHandle matching class_id values from classes configuration
- Define clear, non-overlapping classification categories with descriptive names that help LLM distinguish intent
- Provide classification instructions to guide decision-making for ambiguous cases
- Configure classes with unique IDs (recommend UUIDs) that map to downstream workflow branches
- Consider edge cases: ambiguous questions, out-of-scope inputs, multi-intent queries
- Example classification flow: User query → LLM classifies → Returns class_id "f5660049-284f-41a7-b301-fd24176a711c" → Routes to edge with sourceHandle="f5660049-284f-41a7-b301-fd24176a711c"
- Ideal for: intent routing, topic categorization, customer support triage, multi-branch workflow decision points""",

    NodeType.AGENT: """
Agent nodes execute autonomous reasoning loops with tool calling capabilities using Plugin Agent Strategy architecture.
- Outputs: text (str, final response), usage (dict, token stats with prompt/completion/total), files (array[File]), json (list[dict], ALWAYS list combining agent logs + JSON messages), plus optional custom tool_variables with dynamic names
- json field structure: Always list[dict] = [agent_log1, agent_log2, ..., json_msg1, json_msg2, ...]. If no content: [{"data": []}]
- Agent log dict structure: {id: str (message_id), parent_id: str|None, status: str, data: dict, label: str, metadata: dict, node_id: str, node_execution_id: str, error: str|None}
- Tool variables: Custom outputs defined by agent strategy via VARIABLE messages, accessed as node_id.variable_name, support streaming text or complete values
- Required configuration fields: agent_strategy_provider_name (plugin path), agent_strategy_name (strategy type like "function_calling" or "TOD"), plugin_unique_identifier (version identifier)
- agent_parameters structure: {parameter_name: {type: "constant"|"variable"|"mixed", value: any}}
- Common agent parameters: model (selector with provider/model/completion_params), query (user input, typically sys.query), tools (array[tools] with enabled flag), instruction (agent prompt), storage_key (for conversation memory, typically sys.conversation_id), task_schema (for structured tasks)
- Tool configuration: Each tool has {enabled: bool, provider_name: str, tool_name: str, type: "builtin"|"api"|..., parameters: {param: {auto: 1|2, value: any}}, settings: {setting: {value: any}}, extra: {description: str}}
- Tool parameter forms: auto=1 means LLM auto-generates, auto=2 means manual input (use value field), settings are always manual configuration
- Memory support: Optional memory field with window.size for conversation history context across turns
- Agent strategies include: function_calling (tool orchestration), TOD (task-oriented dialogue with schema-based data collection), react (reasoning+acting), etc.
- Consider: Define clear instruction/task_schema, select minimal necessary tools, set reasonable iteration limits in strategy config, enable memory for multi-turn context
- Ideal for: multi-step autonomous tasks, tool orchestration workflows, conversational data collection, research/analysis agents, dynamic problem-solving
- Example config: model={provider:"langgenius/openai/openai", model:"gpt-4o-mini"}, query={{sys.query}}, tools=[{enabled:true, provider_name:"time", tool_name:"current_time", settings:{timezone:{value:"UTC"}}}], instruction="Your role and task""",

    NodeType.LOOP: """
Loop nodes execute sub-workflows repeatedly for a fixed count with conditional early termination.
- Outputs: dynamic outputs based on loop_variables (e.g., {variable_name: value}), loop_round (int, final iteration number 1-based)
- Loop structure: requires loop-start and loop-end nodes within sub-graph, controlled by loop_id field linking child nodes
- Configuration: loop_count (int, max iterations), break_conditions (list[Condition]), logical_operator ("and"/"or"), loop_variables (list[LoopVariableData]), start_node_id (loop-start node)
- Loop variables: support constant values (value_type="constant") or variable references (value_type="variable"), valid types: string, number, boolean, object, array[string], array[number], array[object], array[boolean]
- Execution flow: 1) Initialize loop variables in variable pool ([node_id, variable_name] selectors), 2) Check break conditions before starting (if true, loop_count=0), 3) Execute sub-graph for each iteration, 4) Check break conditions after each iteration, 5) Update loop variables after each iteration, 6) Terminate on break condition match or loop_count reached
- Break conditions: evaluated via ConditionProcessor with same operators as IF_ELSE node (contains, not contains, start with, end with, is, is not, empty, not empty, in, not in, all of, =, ≠, >, <, ≥, ≤, null, not null, exists, not exists)
- Logical operator: "and" (all conditions must pass), "or" (any condition passes) - evaluated after each iteration
- Output accumulation: "answer" outputs from sub-graph get concatenated with newlines across iterations, other outputs overwrite previous values
- Loop variable updates: after each iteration, loop variables are updated in variable pool and collected into outputs dict with final values at end
- Metadata tracking: loop_duration_map (dict[str, float] with iteration index keys), loop_variable_map (dict[str, dict[str, Any]] with variable snapshots per iteration), completed_reason ("loop_break" or "loop_completed")
- Loop-end node interaction: if loop-end node reached (NodeRunSucceededEvent with NodeType.LOOP_END), sets reach_break_node=True and terminates loop early
- Differs from ITERATION: LOOP is counter-based with fixed max iterations and conditional breaks, ITERATION is array-based processing each element in input array with automatic output flattening
- Sub-graph isolation: each iteration runs in isolated GraphEngine with shared VariablePool, outputs accumulate in parent graph_runtime_state
- Error handling: supports error_strategy and retry_config via BaseLoopNodeData, failures yield LoopFailedEvent with error message
- Use cases: polling with max attempts, retry logic with break conditions, stateful loops tracking variables across iterations, workflows requiring iteration count limits
- Example: loop_count=10, break_conditions=[{variable_selector: ["llm", "success"], comparison_operator: "is", value: true}], logical_operator="or", loop_variables=[{label: "attempt_count", var_type: "number", value_type: "constant", value: 0}]
- Consider: Set reasonable loop_count to prevent excessive iterations, define clear break conditions to avoid unnecessary loops, use loop variables to track state across iterations, handle loop-end node placement for manual early exit
- Ideal for: retry mechanisms with max attempts, polling operations with timeout, stateful iteration workflows, scenarios requiring loop variable tracking and conditional termination""",

    NodeType.LOOP_START: """
Loop start nodes mark the entry point of a loop sub-graph execution.
- Outputs: NONE (no execution logic, immediately succeeds)
- Execution behavior: returns NodeRunResult(status=SUCCEEDED) with no processing
- Purpose: serves as structural marker for loop sub-graph entry point, referenced by parent loop node's start_node_id field
- Graph structure: must be connected as first node in loop sub-graph, all child nodes within loop have loop_id field matching parent loop node ID
- Events: filtered during event streaming (loop_start events not yielded to parent), only internal graph events pass through
- Node data: minimal LoopStartNodeData with only base fields (title, desc, error_strategy, retry_config, default_value_dict)
- Use cases: required structural element for all loop nodes, automatically created by workflow editor when adding loop node
- Not configurable: no custom logic or parameters, purely structural placeholder for loop entry point
- Consider: placement as first node in loop sub-graph, ensure loop_id field on all child nodes matches parent loop node ID""",

    NodeType.LOOP_END: """
Loop end nodes mark a manual early exit point within a loop iteration.
- Outputs: NONE (no execution logic, immediately succeeds)
- Execution behavior: returns NodeRunResult(status=SUCCEEDED), signals parent loop to terminate early via reach_break_node=True
- Early termination: when loop-end node reached during iteration, parent loop breaks immediately after current iteration completes, skipping remaining iterations regardless of loop_count or break_conditions
- Purpose: provides manual early exit mechanism within loop body, allows conditional break logic via preceding if-else nodes
- Graph structure: optional node within loop sub-graph, typically placed after conditional nodes to enable early termination paths
- Detection: parent loop monitors for NodeRunSucceededEvent with node_type==NodeType.LOOP_END, sets reach_break_node flag
- Node data: minimal LoopEndNodeData with only base fields (title, desc, error_strategy, retry_config, default_value_dict)
- Differs from break_conditions: loop-end provides imperative early exit at specific workflow path, break_conditions provide declarative termination evaluated after each iteration
- Use cases: complex conditional early exit logic, multi-branch termination scenarios, manual break triggered by specific workflow paths
- Example pattern: if-else node checks success condition → routes to loop-end on success → parent loop terminates early
- Consider: placement within conditional branches for early exit, combine with if-else for complex break logic, optional node (not required for basic loops)""",

    NodeType.DOCUMENT_EXTRACTOR: """
Document extractor nodes extract text content from 60+ file formats using dual-path detection (MIME type + file extension).
- Outputs (single file): text (str, extracted text content)
- Outputs (file array): text (ArrayStringSegment, list[str] with one string per file)
- Input: variable_selector pointing to FileSegment or ArrayFileSegment from previous nodes
- Detection strategy: tries file extension first if available, falls back to MIME type, raises UnsupportedFileTypeError if both missing
- Document formats (14 types): .txt, .md/.markdown, .html/.htm, .xml, .pdf, .doc, .docx, .csv, .xls/.xlsx, .ppt, .pptx, .epub, .eml, .msg
- Data formats (4 types): .json (pretty-printed with indent=2), .yaml/.yml (dump_all format), .properties (key: value format), .vtt (subtitle with speaker merging)
- Programming languages (30+ extensions): .py, .js, .ts, .jsx, .tsx, .java, .php, .rb, .go, .rs, .swift, .kt, .scala, .sh, .bash, .bat, .ps1, .sql, .r, .m, .pl, .lua, .vim, .asm, .s, .c, .h, .cpp, .hpp, .cc, .cxx, .c++
- Config/style formats (10 types): .css, .scss, .less, .sass, .ini, .cfg, .conf, .toml, .env, .log
- Character encoding: auto-detection via chardet library for all text-based formats, fallback to UTF-8 with errors='ignore'
- Table extraction: .docx/.csv/.xls/.xlsx tables converted to Markdown format (| header | format) with <br> for cell newlines
- External API integration: .doc/.ppt/.pptx/.epub optionally use UNSTRUCTURED_API_URL (configured via dify_config.UNSTRUCTURED_API_URL/UNSTRUCTURED_API_KEY), fallback to local libraries (unstructured, pypandoc)
- Special processing: .vtt merges consecutive utterances by same speaker into Speaker "text" format, .properties converts key=value or key:value to key: value format
- PDF extraction: uses pypdfium2 library with get_text_range() for accurate text extraction
- DOCX extraction: preserves document structure (paragraphs + tables), tables rendered as Markdown with proper row/column alignment
- Error handling: raises TextExtractionError on extraction failures, UnsupportedFileTypeError on unknown formats, FileDownloadError on file access issues
- Consider file size limits (varies by format), encoding ambiguity for non-UTF8 files, table structure complexity for spreadsheets
- Ideal for: document preprocessing pipelines, content analysis workflows, data extraction from file uploads, multi-format text mining""",

    NodeType.LIST_OPERATOR: """
List operator nodes perform filter, extract, order, and limit operations on arrays.
- Outputs: result (array, same type as input), first_record (element or None), last_record (element or None)
- Supports array types: ArrayStringSegment, ArrayNumberSegment, ArrayFileSegment, ArrayBooleanSegment
- Operations applied in order: 1) Filter (conditions), 2) Extract (by serial number), 3) Order (asc/desc), 4) Limit (first N)
- Filter operators by type:
  * String filters (10 operators): contains, not contains, start with, end with, is, is not, empty, not empty, in, not in
  * Number filters (6 operators): =, ≠, <, >, ≤, ≥
  * Boolean filters (2 operators): is, is not
  * File filters: apply string/number/sequence operators on file properties (name, type, extension, mime_type, transfer_method, url, size)
- Filter conditions: multiple conditions are chained sequentially (each condition filters the previous result)
- Extract: 1-based indexing with support for variable templates, result is single-element array, raises error if index < 1 or > array length
- Order: asc/desc sort by value (string/number/boolean arrays) or file properties (name, type, extension, mime_type, transfer_method, url, size for file arrays)
- Limit: take first N elements after filtering/sorting
- Empty input or NoneSegment → result=[] (empty array of same type), first_record=None, last_record=None
- File property filter examples: filter by name (contains ".pdf"), size (> 1000000), type (in ["image", "video"]), extension (is "jpg")
- Ideal for: data filtering, top-K selection, array manipulation, file list processing, conditional list transformations""",

    NodeType.NOTE: """
Note nodes are UI-only annotation elements for workflow documentation and comments.
- Outputs: NONE (no execution, filtered before workflow runs)
- Type value: Empty string ("") in DSL YAML files, NodeType.NOTE enum in Python models
- UI representation: Appears as "custom-note" type in frontend graph rendering
- Execution behavior: Explicitly filtered out during graph construction (graph.py line 290), never reaches node_mapping or execution phase
- NOT present in official Dify NodeType enum, exists only in DSL layer for workflow documentation
- Configuration fields: title (str), theme (blue/cyan/green/yellow/pink/violet), text (str, plain text content), author (str), showAuthor (bool), width (int, default 240), height (int, default 88)
- Rich text support: data field contains Lexical Editor JSON structure with formatted content (paragraphs, code blocks, lists, styling)
- Lexical Editor format: {root: {children: [{type: "paragraph", children: [{type: "text", text: "...", format: 0}]}], direction: "ltr", version: 1}}
- Use cases: workflow logic explanations, decision point rationale, integration notes, TODO comments, code examples, complex configuration documentation
- Best practices: Add notes at complex branching points, document non-obvious variable transformations, explain external API integration details, provide usage examples for template variables
- Consider placement: Position notes near relevant nodes for context, use descriptive titles for quick scanning, leverage rich text formatting for code snippets and structured explanations
- Maintenance value: Improves workflow understandability for team collaboration, reduces cognitive load when revisiting workflows after time, serves as inline documentation without affecting execution performance""",
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