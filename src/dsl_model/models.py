"""
Elegant Dify Workflow DSL Models

Inspired by Dify's official architecture with clean inheritance hierarchy,
strict typing, and comprehensive field validation using pure Pydantic models.

Key Design Principles:
1. Inheritance-based node type system (BaseNodeData -> Specific Node Types)
2. Strict typing with proper validation
3. Discriminated unions for type safety
4. Clean separation of concerns
5. Comprehensive field validation with sensible defaults
"""

from abc import ABC
from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from dsl_model.dsl import DifyWorkflowDSL


# ============================================================================
# Core Enums
# ============================================================================

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


# ============================================================================
# Core Data Structures
# ============================================================================

class VariableSelector(BaseModel):
    """Variable selector for referencing node outputs"""
    variable: str
    value_selector: Sequence[str]


class Position(BaseModel):
    """Node position on canvas"""
    x: float
    y: float


class Viewport(BaseModel):
    """Canvas viewport configuration"""
    x: float
    y: float
    zoom: float = Field(gt=0.0, default=1.0)


class Variable(BaseModel):
    """Workflow input variable definition"""
    variable: str = Field(min_length=1)
    label: str = Field(min_length=1)
    type: VariableType
    required: bool = Field(default=False)
    max_length: Optional[int] = Field(default=None, ge=1)
    options: List[str] = Field(default_factory=list)

    @field_validator('max_length')
    @classmethod
    def validate_max_length(cls, v, info):
        # More lenient validation for DSL compatibility
        if v is not None and v < 0:
            raise ValueError('max_length must be positive')
        return v


class ModelConfig(BaseModel):
    """LLM model configuration"""
    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)
    mode: LLMMode
    completion_params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('completion_params')
    @classmethod
    def validate_completion_params(cls, v):
        """Validate common completion parameters"""
        allowed_params = {
            'temperature', 'top_p', 'top_k', 'max_tokens', 
            'presence_penalty', 'frequency_penalty', 'response_format'
        }
        for key in v:
            if key not in allowed_params:
                continue
            # Validate specific parameter ranges
            if key == 'temperature' and not (0 <= v[key] <= 2):
                raise ValueError('temperature must be between 0 and 2')
            elif key == 'top_p' and not (0 <= v[key] <= 1):
                raise ValueError('top_p must be between 0 and 1')
            elif key in ['presence_penalty', 'frequency_penalty'] and not (-2 <= v[key] <= 2):
                raise ValueError(f'{key} must be between -2 and 2')
        return v


class PromptMessage(BaseModel):
    """LLM prompt message"""
    role: Literal["system", "user", "assistant"]
    text: str = Field(default="")
    id: Optional[str] = None

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('prompt message text cannot be empty')
        return v


class ContextConfig(BaseModel):
    """Context configuration for LLM nodes"""
    enabled: bool = Field(default=False)
    variable_selector: Optional[List[str]] = None

    @model_validator(mode='after')
    def validate_context_config(self):
        if self.enabled and not self.variable_selector:
            raise ValueError('variable_selector is required when context is enabled')
        return self


class VisionConfig(BaseModel):
    """Vision configuration for LLM nodes"""
    enabled: bool = Field(default=False)
    variable_selector: Sequence[str] = Field(default_factory=lambda: ["sys", "files"])
    detail: Literal["low", "high"] = Field(default="high")

    @model_validator(mode='after')
    def validate_vision_config(self):
        if self.enabled and not self.variable_selector:
            raise ValueError('variable_selector is required when vision is enabled')
        return self


# ============================================================================
# Base Node Data (Abstract)
# ============================================================================

class BaseNodeData(BaseModel, ABC):
    """Abstract base class for all node data types"""
    
    type: NodeType = Field(description="Node type identifier")
    title: str = Field(min_length=1, description="Node display title")
    desc: str = Field(default="", description="Node description")
    version: str = Field(default="1", description="Node version")
    error_strategy: Optional[ErrorStrategy] = Field(default=None)
    
    # Allow extra fields for DSL compatibility
    model_config = ConfigDict(extra="allow")

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('title cannot be empty')
        return v.strip()


# ============================================================================
# Specific Node Data Types
# ============================================================================

class StartNodeData(BaseNodeData):
    """Start node - workflow entry point"""
    type: Literal[NodeType.START] = NodeType.START
    variables: List[Variable] = Field(default_factory=list)


class EndNodeData(BaseNodeData):
    """End node - workflow termination"""
    type: Literal[NodeType.END] = NodeType.END


class AnswerNodeData(BaseNodeData):
    """Answer node - provides response in chat mode"""
    type: Literal[NodeType.ANSWER] = NodeType.ANSWER
    answer: str = Field(description="Answer template with variable substitution")

    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v):
        if not v.strip():
            raise ValueError('answer cannot be empty')
        return v


class LLMNodeData(BaseNodeData):
    """LLM node - AI model interaction"""
    type: Literal[NodeType.LLM] = NodeType.LLM
    model: ModelConfig
    prompt_template: List[PromptMessage]
    context: Optional[ContextConfig] = None
    vision: Optional[VisionConfig] = None
    memory: Optional[Dict[str, Any]] = None
    structured_output: Optional[Dict[str, Any]] = None
    structured_output_enabled: bool = Field(default=False)

    @field_validator('prompt_template')
    @classmethod
    def validate_prompt_template(cls, v):
        if not v:
            raise ValueError('prompt_template cannot be empty')
        # Ensure at least one user message
        has_user = any(msg.role == "user" for msg in v)
        if not has_user:
            raise ValueError('prompt_template must contain at least one user message')
        return v

    @model_validator(mode='after')
    def validate_structured_output(self):
        if self.structured_output_enabled and not self.structured_output:
            raise ValueError('structured_output is required when structured_output_enabled is True')
        return self


class CodeNodeData(BaseNodeData):
    """Code node - execute Python/JavaScript code"""
    
    class Output(BaseModel):
        """Code node output definition"""
        type: SegmentType
        children: Optional[Dict[str, "CodeNodeData.Output"]] = None

        @field_validator('type')
        @classmethod
        def validate_output_type(cls, v):
            allowed_types = {
                SegmentType.STRING, SegmentType.NUMBER, SegmentType.OBJECT, SegmentType.BOOLEAN,
                SegmentType.ARRAY_STRING, SegmentType.ARRAY_NUMBER, SegmentType.ARRAY_OBJECT,
                SegmentType.ARRAY_BOOLEAN, SegmentType.ARRAY_FILE
            }
            if v not in allowed_types:
                raise ValueError(f'Invalid output type: {v}')
            return v

    class Dependency(BaseModel):
        """Code node dependency"""
        name: str = Field(min_length=1)
        version: str = Field(min_length=1)

    type: Literal[NodeType.CODE] = NodeType.CODE
    code_language: CodeLanguage
    code: str = Field(min_length=1)
    variables: List[VariableSelector] = Field(default_factory=list)
    outputs: Dict[str, Output]
    dependencies: List[Dependency] = Field(default_factory=list)

    @field_validator('code')
    @classmethod
    def validate_code(cls, v):
        if not v.strip():
            raise ValueError('code cannot be empty')
        return v

    @field_validator('outputs')
    @classmethod
    def validate_outputs(cls, v):
        if not v:
            raise ValueError('at least one output must be defined')
        return v


class HTTPRequestNodeData(BaseNodeData):
    """HTTP request node"""
    
    class Authorization(BaseModel):
        """HTTP authorization configuration"""
        type: Literal["no-auth", "api-key", "bearer-token"] = "no-auth"
        config: Optional[Dict[str, str]] = None

    type: Literal[NodeType.HTTP_REQUEST] = NodeType.HTTP_REQUEST
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    url: str = Field(min_length=1, description="Request URL")
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    authorization: Authorization = Field(default_factory=Authorization)
    timeout: int = Field(default=30, ge=1, le=300)

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://', '{{')):
            raise ValueError('URL must be a valid HTTP(S) URL or template variable')
        return v


class ToolNodeData(BaseNodeData):
    """Tool node - external tool integration"""
    
    class ParameterSchema(BaseModel):
        """Tool parameter schema"""
        name: str = Field(min_length=1)
        type: str = Field(min_length=1)
        required: bool = Field(default=False)
        description: str = Field(default="")

    type: Literal[NodeType.TOOL] = NodeType.TOOL
    provider_id: str = Field(min_length=1)
    provider_name: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_label: str = Field(default="")
    tool_parameters: Dict[str, Any] = Field(default_factory=dict)
    tool_configurations: Dict[str, Any] = Field(default_factory=dict)


class IfElseNodeData(BaseNodeData):
    """If-else node - conditional branching"""
    
    class Condition(BaseModel):
        """Single condition"""
        variable_selector: List[str] = Field(min_items=1)
        comparison_operator: Literal["=", "≠", ">", "<", "≥", "≤", "contains", "starts with", "ends with"]
        value: str

    class Case(BaseModel):
        """Condition case"""
        case_id: str = Field(min_length=1)
        conditions: List["IfElseNodeData.Condition"] = Field(min_items=1)
        logical_operator: Literal["and", "or"] = "and"

    type: Literal[NodeType.IF_ELSE] = NodeType.IF_ELSE
    cases: List[Case] = Field(min_items=1)

    @field_validator('cases')
    @classmethod
    def validate_cases(cls, v):
        case_ids = [case.case_id for case in v]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError('case_id must be unique')
        return v


class TemplateTransformNodeData(BaseNodeData):
    """Template transform node - text template processing"""
    type: Literal[NodeType.TEMPLATE_TRANSFORM] = NodeType.TEMPLATE_TRANSFORM
    template: str = Field(min_length=1, description="Jinja2 template")

    @field_validator('template')
    @classmethod
    def validate_template(cls, v):
        if not v.strip():
            raise ValueError('template cannot be empty')
        return v


class VariableAssignerNodeData(BaseNodeData):
    """Variable assigner node"""
    type: Literal[NodeType.VARIABLE_ASSIGNER] = NodeType.VARIABLE_ASSIGNER
    assigned_variable_selector: List[str] = Field(min_items=1)
    input_variable_selector: List[str] = Field(min_items=1)
    write_mode: Literal["over-write", "append", "clear"] = "over-write"


class KnowledgeRetrievalNodeData(BaseNodeData):
    """Knowledge retrieval node"""
    type: Literal[NodeType.KNOWLEDGE_RETRIEVAL] = NodeType.KNOWLEDGE_RETRIEVAL
    dataset_ids: List[str] = Field(min_items=1)
    query_variable_selector: List[str] = Field(min_items=1)
    retrieval_mode: Literal["single", "multiple"] = "single"
    top_k: int = Field(default=3, ge=1, le=20)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentNodeData(BaseNodeData):
    """Agent node - intelligent agent interaction"""
    type: Literal[NodeType.AGENT] = NodeType.AGENT
    agent_strategy: str = Field(min_length=1)
    agent_parameters: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None


class IterationNodeData(BaseNodeData):
    """Iteration node - loop processing"""
    type: Literal[NodeType.ITERATION] = NodeType.ITERATION
    iterator_selector: List[str] = Field(min_items=1)
    output_selector: List[str] = Field(min_items=1)
    output_type: SegmentType = SegmentType.ARRAY_OBJECT


class ParameterExtractorNodeData(BaseNodeData):
    """Parameter extractor node"""
    type: Literal[NodeType.PARAMETER_EXTRACTOR] = NodeType.PARAMETER_EXTRACTOR
    query: str = Field(min_length=1)
    parameters: List[Dict[str, Any]] = Field(min_items=1)


class QuestionClassifierNodeData(BaseNodeData):
    """Question classifier node"""
    type: Literal[NodeType.QUESTION_CLASSIFIER] = NodeType.QUESTION_CLASSIFIER
    query_variable_selector: List[str] = Field(min_items=1)
    classes: List[Dict[str, Any]] = Field(min_items=2)


# ============================================================================
# Node Data Union with Discriminator
# ============================================================================

# For DSL compatibility, use a flexible base type with model_rebuild
class FlexibleNodeData(BaseModel):
    """Flexible node data for DSL compatibility"""
    type: str = Field(description="Node type identifier")
    title: str = Field(default="", description="Node display title") 
    desc: str = Field(default="", description="Node description")
    
    model_config = ConfigDict(extra="allow")

# Union of all node data types - prioritize most specific types first
NodeData = Union[
    StartNodeData,
    EndNodeData,
    AnswerNodeData,
    LLMNodeData,
    CodeNodeData,
    HTTPRequestNodeData,
    ToolNodeData,
    IfElseNodeData,
    TemplateTransformNodeData,
    VariableAssignerNodeData,
    KnowledgeRetrievalNodeData,
    AgentNodeData,
    IterationNodeData,
    ParameterExtractorNodeData,
    QuestionClassifierNodeData,
    FlexibleNodeData,  # Fallback for unknown types
]


# ============================================================================
# Graph Structure
# ============================================================================

class EdgeData(BaseModel):
    """Edge connection metadata"""
    sourceType: str = Field(min_length=1)
    targetType: str = Field(min_length=1)
    isInIteration: bool = Field(default=False)
    isInLoop: bool = Field(default=False)


class Edge(BaseModel):
    """Graph edge definition"""
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    sourceHandle: str = Field(default="source")
    targetHandle: str = Field(default="target")
    type: str = Field(default="custom")
    data: EdgeData
    zIndex: int = Field(default=0)

    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not v.strip():
            raise ValueError('edge id cannot be empty')
        return v


class Node(BaseModel):
    """Graph node definition"""
    id: str = Field(min_length=1)
    data: NodeData
    position: Position
    positionAbsolute: Position
    height: int = Field(default=89, ge=1)
    width: int = Field(default=244, ge=1)
    selected: bool = Field(default=False)
    sourcePosition: Literal["left", "right", "top", "bottom"] = Field(default="right")
    targetPosition: Literal["left", "right", "top", "bottom"] = Field(default="left")
    type: str = Field(default="custom")

    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not v.strip():
            raise ValueError('node id cannot be empty')
        return v

    @model_validator(mode='after')
    def validate_node_consistency(self):
        """Ensure node data type matches expected structure"""
        if hasattr(self.data, 'type'):
            # Additional validation can be added here
            pass
        return self


class Graph(BaseModel):
    """Complete workflow graph"""
    nodes: List[Node] = Field(min_items=1)
    edges: List[Edge] = Field(default_factory=list)
    viewport: Viewport = Field(default_factory=Viewport)

    @model_validator(mode='after')
    def validate_graph_structure(self):
        """Validate graph connectivity and structure"""
        node_ids = {node.id for node in self.nodes}
        
        # Validate edges reference valid nodes
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f'Edge source "{edge.source}" references non-existent node')
            if edge.target not in node_ids:
                raise ValueError(f'Edge target "{edge.target}" references non-existent node')
        
        # Ensure there's exactly one start node
        start_nodes = [node for node in self.nodes if node.data.type == NodeType.START]
        if len(start_nodes) != 1:
            raise ValueError('Graph must have exactly one start node')
        
        return self


# ============================================================================
# Workflow Features and Configuration
# ============================================================================

class FileUploadConfig(BaseModel):
    """File upload configuration"""
    audio_file_size_limit: int = Field(default=50, ge=1)
    batch_count_limit: int = Field(default=5, ge=1)
    file_size_limit: int = Field(default=15, ge=1)
    image_file_size_limit: int = Field(default=10, ge=1)
    video_file_size_limit: int = Field(default=100, ge=1)
    workflow_file_upload_limit: int = Field(default=10, ge=1)


class FileUploadFeature(BaseModel):
    """File upload feature configuration"""
    enabled: bool = Field(default=False)
    allowed_file_extensions: List[str] = Field(default_factory=list)
    allowed_file_types: List[str] = Field(default_factory=list)
    allowed_file_upload_methods: List[str] = Field(
        default_factory=lambda: ["local_file", "remote_url"]
    )
    fileUploadConfig: FileUploadConfig = Field(default_factory=FileUploadConfig)
    number_limits: int = Field(default=3, ge=1)


class WorkflowFeatures(BaseModel):
    """Workflow features configuration"""
    file_upload: FileUploadFeature = Field(default_factory=FileUploadFeature)
    opening_statement: str = Field(default="")
    retriever_resource: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True}
    )
    sensitive_word_avoidance: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False}
    )
    speech_to_text: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False}
    )
    suggested_questions: List[str] = Field(default_factory=list)
    suggested_questions_after_answer: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False}
    )
    text_to_speech: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "language": "", "voice": ""}
    )
    
    model_config = ConfigDict(extra="allow")


class Workflow(BaseModel):
    """Complete workflow definition"""
    conversation_variables: List[Any] = Field(default_factory=list)
    environment_variables: List[Any] = Field(default_factory=list)
    features: WorkflowFeatures = Field(default_factory=WorkflowFeatures)
    graph: Graph


# ============================================================================
# Application Configuration
# ============================================================================

class AppMetadata(BaseModel):
    """Application metadata"""
    description: str = Field(default="")
    icon: str = Field(default="🤖")
    icon_background: str = Field(default="#FFEAD5", pattern=r"^#[0-9A-Fa-f]{6}$")
    mode: WorkflowMode
    name: str = Field(min_length=1)
    use_icon_as_answer_icon: bool = Field(default=False)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('app name cannot be empty')
        return v.strip()


class Dependency(BaseModel):
    """External dependency definition"""
    current_identifier: Optional[str] = None
    type: str = Field(min_length=1)
    value: Dict[str, Any] = Field(description="Dependency configuration")


# ============================================================================
# Root DSL Model
# ============================================================================


# ============================================================================
# Utility Functions
# ============================================================================

def create_node(
    node_id: str,
    node_type: NodeType,
    title: str,
    position: Position,
    **kwargs
) -> Node:
    """Create a properly typed node with validation"""
    
    # Map node type to data class
    node_data_map = {
        NodeType.START: StartNodeData,
        NodeType.END: EndNodeData,
        NodeType.ANSWER: AnswerNodeData,
        NodeType.LLM: LLMNodeData,
        NodeType.CODE: CodeNodeData,
        NodeType.HTTP_REQUEST: HTTPRequestNodeData,
        NodeType.TOOL: ToolNodeData,
        NodeType.IF_ELSE: IfElseNodeData,
        NodeType.TEMPLATE_TRANSFORM: TemplateTransformNodeData,
        NodeType.VARIABLE_ASSIGNER: VariableAssignerNodeData,
        NodeType.KNOWLEDGE_RETRIEVAL: KnowledgeRetrievalNodeData,
        NodeType.AGENT: AgentNodeData,
        NodeType.ITERATION: IterationNodeData,
        NodeType.PARAMETER_EXTRACTOR: ParameterExtractorNodeData,
        NodeType.QUESTION_CLASSIFIER: QuestionClassifierNodeData,
    }
    
    data_class = node_data_map.get(node_type)
    if not data_class:
        raise ValueError(f"Unsupported node type: {node_type}")
    
    # Create node data with proper type and validation
    node_data = data_class(
        type=node_type,
        title=title,
        **kwargs
    )
    
    return Node(
        id=node_id,
        data=node_data,
        position=position,
        positionAbsolute=position
    )


def create_edge(
    edge_id: str,
    source_id: str,
    target_id: str,
    source_type: str,
    target_type: str
) -> Edge:
    """Create a validated edge"""
    return Edge(
        id=edge_id,
        source=source_id,
        target=target_id,
        data=EdgeData(
            sourceType=source_type,
            targetType=target_type
        )
    )


def validate_dsl(data: Dict[str, Any]) -> DifyWorkflowDSL:
    """Validate DSL data and return parsed workflow"""
    try:
        return DifyWorkflowDSL.model_validate(data)
    except Exception as e:
        raise ValueError(f"DSL validation failed: {e}")