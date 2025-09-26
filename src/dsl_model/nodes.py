from typing import Any, Dict, List, Literal, Optional, Sequence, Union, Annotated

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .enums import NodeType, SegmentType, CodeLanguage
from .core import (
    BaseNodeData,
    ContextConfig,
    LLMNodeChatModelMessage,
    LLMNodeCompletionModelPromptTemplate,
    ModelConfig,
    PromptConfig,
    PromptMessage,
    Variable,
    VariableSelector,
    VisionConfig,
)


class StartNodeData(BaseNodeData):
    """Start node - workflow entry point"""
    type: Literal[NodeType.START] = Field(default=NodeType.START)
    variables: List[Variable] = Field(default_factory=list)


class EndNodeData(BaseNodeData):
    """End node - workflow termination"""
    type: Literal[NodeType.END] = Field(default=NodeType.END)


class AnswerNodeData(BaseNodeData):
    """Answer node - provides response in chat mode"""
    type: Literal[NodeType.ANSWER] = Field(default=NodeType.ANSWER)
    answer: str = Field(description="Answer template with variable substitution")


class LLMNodeData(BaseNodeData):
    """LLM node - AI model interaction"""
    type: Literal[NodeType.LLM] = Field(default=NodeType.LLM)
    model: ModelConfig
    prompt_template: Union[Sequence[LLMNodeChatModelMessage], LLMNodeCompletionModelPromptTemplate]
    prompt_config: PromptConfig = Field(default_factory=PromptConfig)
    memory: Optional[Dict[str, Any]] = None
    context: ContextConfig
    vision: VisionConfig = Field(default_factory=VisionConfig)
    structured_output: Optional[Dict[str, Any]] = None
    structured_output_switch_on: bool = Field(default=False, alias="structured_output_enabled")
    reasoning_format: Literal["separated", "tagged"] = Field(default="tagged")

    @property
    def structured_output_enabled(self) -> bool:
        """Backward compatibility property"""
        return self.structured_output_switch_on and self.structured_output is not None


class CodeNodeData(BaseNodeData):
    """Code node - execute Python/JavaScript code"""
    type: Literal[NodeType.CODE] = Field(default=NodeType.CODE)

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
        name: str
        version: str

    variables: List[VariableSelector]
    code_language: CodeLanguage
    code: str
    outputs: Dict[str, Output]
    dependencies: Optional[List[Dependency]] = None


class HTTPRequestNodeData(BaseNodeData):
    """HTTP request node"""
    type: Literal[NodeType.HTTP_REQUEST] = Field(default=NodeType.HTTP_REQUEST)

    class Authorization(BaseModel):
        """HTTP authorization configuration"""
        type: Literal["no-auth", "api-key", "bearer-token"] = "no-auth"
        config: Optional[Dict[str, str]] = None

    class Timeout(BaseModel):
        """HTTP timeout configuration matching Dify's HttpRequestNodeTimeout"""
        connect: int = 10
        read: int = 30
        write: int = 30

    method: str
    url: str = Field(min_length=1, description="Request URL")
    headers: str
    params: str
    body: Optional[Dict[str, Any]] = None
    authorization: Authorization = Field(default_factory=Authorization)
    timeout: Optional[Timeout] = None

    model_config = ConfigDict(extra="allow")


class ToolNodeData(BaseNodeData):
    """Tool node - external tool integration"""
    type: Literal[NodeType.TOOL] = Field(default=NodeType.TOOL)

    class ParameterSchema(BaseModel):
        """Tool parameter schema"""
        name: str = Field(min_length=1)
        type: str = Field(min_length=1)
        required: bool = Field(default=False)
        description: str = Field(default="")

    # Removed type field to match dify/api
    provider_id: str = Field(min_length=1)
    provider_name: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_label: str = Field(default="")
    tool_parameters: Dict[str, Any] = Field(default_factory=dict)
    tool_configurations: Dict[str, Any] = Field(default_factory=dict)


class IfElseNodeData(BaseNodeData):
    """If-else node - conditional branching"""
    type: Literal[NodeType.IF_ELSE] = Field(default=NodeType.IF_ELSE)

    class SubCondition(BaseModel):
        """Sub-condition for complex conditions"""
        key: str
        comparison_operator: Literal[
            # for string or array
            "contains", "not contains", "start with", "end with",
            "is", "is not", "empty", "not empty", "in", "not in", "all of",
            # for number
            "=", "≠", ">", "<", "≥", "≤", "null", "not null",
            # for file
            "exists", "not exists"
        ]
        value: Optional[Union[str, List[str]]] = None

    class SubVariableCondition(BaseModel):
        """Sub-variable condition for complex conditions"""
        logical_operator: Literal["and", "or"]
        conditions: List["IfElseNodeData.SubCondition"] = Field(default_factory=list)

    class Condition(BaseModel):
        """Single condition"""
        id: Optional[str] = Field(default=None, description="Condition ID")
        variable_selector: List[str]
        comparison_operator: Literal[
            # for string or array
            "contains", "not contains", "start with", "end with",
            "is", "is not", "empty", "not empty", "in", "not in", "all of",
            # for number
            "=", "≠", ">", "<", "≥", "≤", "null", "not null",
            # for file
            "exists", "not exists"
        ]
        value: Optional[Union[str, List[str], bool]] = None
        varType: Optional[str] = Field(default=None, description="Variable type (e.g., 'string')")
        sub_variable_condition: Optional["IfElseNodeData.SubVariableCondition"] = None

    class Case(BaseModel):
        """Condition case"""
        case_id: str = Field(min_length=1)
        id: Optional[str] = Field(default=None, description="Case ID (often same as case_id)")
        conditions: List["IfElseNodeData.Condition"]
        logical_operator: Literal["and", "or"] = Field(default="and")

    # Deprecated top-level fields for backward compatibility
    logical_operator: Optional[Literal["and", "or"]] = Field(default="and", deprecated=True)
    conditions: Optional[List[Condition]] = Field(default=None, deprecated=True)

    # Main cases field
    cases: Optional[List[Case]] = None


class TemplateTransformNodeData(BaseNodeData):
    """Template transform node - text template processing"""
    type: Literal[NodeType.TEMPLATE_TRANSFORM] = Field(default=NodeType.TEMPLATE_TRANSFORM)
    template: str = Field(min_length=1, description="Jinja2 template")
    variables: Sequence[VariableSelector]



class VariableAssignerNodeData(BaseNodeData):
    """Variable assigner node"""
    type: Literal[NodeType.VARIABLE_ASSIGNER, NodeType.LEGACY_VARIABLE_AGGREGATOR] = Field(default=NodeType.VARIABLE_ASSIGNER)
    assigned_variable_selector: Optional[List[str]] = None
    input_variable_selector: Optional[List[str]] = None
    write_mode: Literal["over-write", "append", "clear"] = "over-write"


    model_config = ConfigDict(extra="allow")


class KnowledgeRetrievalNodeData(BaseNodeData):
    """Knowledge retrieval node"""
    type: Literal[NodeType.KNOWLEDGE_RETRIEVAL] = Field(default=NodeType.KNOWLEDGE_RETRIEVAL)
    dataset_ids: List[str]
    query_variable_selector: List[str]
    retrieval_mode: Literal["single", "multiple"] = "single"
    top_k: int = Field(default=3, ge=1, le=20)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentNodeData(BaseNodeData):
    """Agent node - intelligent agent interaction"""
    type: Literal[NodeType.AGENT] = Field(default=NodeType.AGENT)
    agent_strategy: Optional[str] = None
    agent_parameters: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class IterationNodeData(BaseNodeData):
    """Iteration node - loop processing"""
    type: Literal[NodeType.ITERATION] = Field(default=NodeType.ITERATION)
    iterator_selector: List[str]
    output_selector: List[str]
    output_type: SegmentType = SegmentType.ARRAY_OBJECT


class ParameterExtractorNodeData(BaseNodeData):
    """Parameter extractor node"""
    type: Literal[NodeType.PARAMETER_EXTRACTOR] = Field(default=NodeType.PARAMETER_EXTRACTOR)
    query: Union[str, List[str]] = Field(description="Query string or list of query strings")
    parameters: List[Dict[str, Any]]
    model: Optional[ModelConfig] = None
    instruction: Optional[str] = None
    reasoning_mode: Literal["function_call", "prompt"] = "function_call"
    vision: Optional[VisionConfig] = None


class QuestionClassifierNodeData(BaseNodeData):
    """Question classifier node"""
    type: Literal[NodeType.QUESTION_CLASSIFIER] = Field(default=NodeType.QUESTION_CLASSIFIER)
    query_variable_selector: List[str]
    classes: List[Dict[str, Any]]


class IterationStartNodeData(BaseNodeData):
    """Iteration start pseudo node used by Dify to mark loop entry"""
    type: Literal[NodeType.ITERATION_START] = Field(default=NodeType.ITERATION_START)
    model_config = ConfigDict(extra="allow")


class LoopStartNodeData(BaseNodeData):
    """Loop start pseudo node"""
    type: Literal[NodeType.LOOP_START] = Field(default=NodeType.LOOP_START)
    model_config = ConfigDict(extra="allow")


class LoopEndNodeData(BaseNodeData):
    """Loop end pseudo node"""
    type: Literal[NodeType.LOOP_END] = Field(default=NodeType.LOOP_END)
    model_config = ConfigDict(extra="allow")


class VariableAggregatorNodeData(BaseNodeData):
    """Variable aggregator node - aggregate values across iterations or lists"""
    type: Literal[NodeType.VARIABLE_AGGREGATOR] = Field(default=NodeType.VARIABLE_AGGREGATOR)
    # Keep permissive; different DSLs use various shapes
    aggregator: Optional[str] = None
    source_selector: Optional[List[str]] = None
    model_config = ConfigDict(extra="allow")


class DocumentExtractorNodeData(BaseNodeData):
    """Document extractor node - extract text from documents"""
    type: Literal[NodeType.DOCUMENT_EXTRACTOR] = Field(default=NodeType.DOCUMENT_EXTRACTOR)
    variable_selector: List[str]
    model_config = ConfigDict(extra="allow")


class ListOperatorNodeData(BaseNodeData):
    """List operator node - perform operations on lists"""
    type: Literal[NodeType.LIST_OPERATOR] = Field(default=NodeType.LIST_OPERATOR)

    class FilterCondition(BaseModel):
        """Filter condition for list operations"""
        variable_selector: List[str]
        filter_operator: Literal[
            "contains", "start with", "end with", "is", "in", "empty",
            "not contains", "is not", "not in", "not empty",
            "=", "≠", "<", ">", "≥", "≤"
        ]
        value: str = ""

    operation: Literal["filter", "map", "extract"] = "filter"
    variable_selector: Optional[List[str]] = None
    filter_conditions: Optional[List[FilterCondition]] = None
    model_config = ConfigDict(extra="allow")


class NoteNodeData(BaseModel):
    """Note node - annotation/comment node for workflow documentation"""
    type: Literal[NodeType.NOTE] = Field(default=NodeType.NOTE)
    title: str = Field(default="")
    theme: Literal["blue", "cyan", "green", "yellow", "pink", "violet"]
    text: Optional[str] = None
    author: str = Field(default="")
    showAuthor: bool = Field(default=False)
    width: int = Field(default=240)
    height: int = Field(default=88)
    # Note nodes can have complex rich text data structure
    data: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra="allow")

# Discriminated union of all node data types by the 'type' field
NodeData = Annotated[
    Union[
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
        IterationStartNodeData,
        LoopStartNodeData,
        LoopEndNodeData,
        VariableAggregatorNodeData,
        ParameterExtractorNodeData,
        QuestionClassifierNodeData,
        DocumentExtractorNodeData,
        ListOperatorNodeData,
        NoteNodeData,
    ],
    Field(discriminator='type')
]
