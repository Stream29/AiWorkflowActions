from enum import StrEnum


class NodeType(StrEnum):
    """All supported node types"""
    START = "start"
    END = "end"
    ANSWER = "answer"
    LLM = "llm"
    AGENT = "agent"
    CODE = "code"
    IF_ELSE = "if-else"
    TEMPLATE_TRANSFORM = "template-transform"
    PARAMETER_EXTRACTOR = "parameter-extractor"
    VARIABLE_ASSIGNER = "assigner"
    VARIABLE_AGGREGATOR = "variable-aggregator"
    ITERATION = "iteration"
    ITERATION_START = "iteration-start"
    HTTP_REQUEST = "http-request"
    TOOL = "tool"
    KNOWLEDGE_RETRIEVAL = "knowledge-retrieval"
    QUESTION_CLASSIFIER = "question-classifier"
    DOCUMENT_EXTRACTOR = "document-extractor"
    LIST_OPERATOR = "list-operator"
    CUSTOM = "custom"


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
