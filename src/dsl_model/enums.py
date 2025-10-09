from enum import StrEnum


class NodeType(StrEnum):
    """All supported node types"""
    START = "start"
    END = "end"
    ANSWER = "answer"
    LLM = "llm"
    KNOWLEDGE_RETRIEVAL = "knowledge-retrieval"
    IF_ELSE = "if-else"
    CODE = "code"
    TEMPLATE_TRANSFORM = "template-transform"
    QUESTION_CLASSIFIER = "question-classifier"
    HTTP_REQUEST = "http-request"
    TOOL = "tool"
    VARIABLE_AGGREGATOR = "variable-aggregator"
    LEGACY_VARIABLE_AGGREGATOR = "variable-assigner"  # For backward compatibility
    LOOP = "loop"
    LOOP_START = "loop-start"
    LOOP_END = "loop-end"
    ITERATION = "iteration"
    ITERATION_START = "iteration-start"  # Fake start node for iteration
    PARAMETER_EXTRACTOR = "parameter-extractor"
    VARIABLE_ASSIGNER = "assigner"
    DOCUMENT_EXTRACTOR = "document-extractor"
    LIST_OPERATOR = "list-operator"
    AGENT = "agent"
    NOTE = ""  # Note/comment node with empty type value


class VariableType(StrEnum):
    """Variable types for workflow inputs"""
    TEXT_INPUT = "text-input"
    PARAGRAPH = "paragraph"
    SELECT = "select"
    NUMBER = "number"
    FILE = "file"
    FILES = "files"


class WorkflowMode(StrEnum):
    """Workflow execution modes"""
    WORKFLOW = "workflow"
    ADVANCED_CHAT = "advanced-chat"
    CHAT = "chat"
    COMPLETION = "completion"


class ErrorStrategy(StrEnum):
    """Error handling strategies"""
    FAIL_BRANCH = "fail-branch"
    DEFAULT_VALUE = "default-value"


class LLMMode(StrEnum):
    """LLM operation modes"""
    CHAT = "chat"
    COMPLETION = "completion"


class CodeLanguage(StrEnum):
    """Supported code execution languages"""
    PYTHON3 = "python3"
    JAVASCRIPT = "javascript"


class SegmentType(StrEnum):
    """Variable segment types"""
    STRING = "string"
    NUMBER = "number"
    OBJECT = "object"
    BOOLEAN = "boolean"
    ARRAY_STRING = "array[string]"
    ARRAY_NUMBER = "array[number]"
    ARRAY_OBJECT = "array[object]"
    ARRAY_BOOLEAN = "array[boolean]"
    ARRAY_FILE = "array[file]"


class DefaultValueType(StrEnum):
    """Default value types for error handling"""
    STRING = "string"
    NUMBER = "number"
    OBJECT = "object"
    BOOLEAN = "boolean"
    ARRAY_NUMBER = "array[number]"
    ARRAY_STRING = "array[string]"
    ARRAY_OBJECT = "array[object]"
    ARRAY_BOOLEAN = "array[boolean]"
    ARRAY_FILES = "array[file]"
