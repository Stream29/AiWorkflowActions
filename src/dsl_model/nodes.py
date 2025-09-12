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
    type: Literal["start"] = Field(default="start")
    variables: List[Variable] = Field(default_factory=list)


class EndNodeData(BaseNodeData):
    """End node - workflow termination"""
    type: Literal["end"] = Field(default="end")


class AnswerNodeData(BaseNodeData):
    """Answer node - provides response in chat mode"""
    type: Literal["answer"] = Field(default="answer")
    answer: str = Field(description="Answer template with variable substitution")

    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v):
        if not v.strip():
            raise ValueError('answer cannot be empty')
        return v


class LLMNodeData(BaseNodeData):
    """LLM node - AI model interaction"""
    type: Literal["llm"] = Field(default="llm")
    model: ModelConfig
    prompt_template: Union[Sequence[LLMNodeChatModelMessage], LLMNodeCompletionModelPromptTemplate]
    prompt_config: PromptConfig = Field(default_factory=PromptConfig)
    memory: Optional[Dict[str, Any]] = None
    context: ContextConfig
    vision: VisionConfig = Field(default_factory=VisionConfig)
    structured_output: Optional[Dict[str, Any]] = None
    structured_output_switch_on: bool = Field(default=False, alias="structured_output_enabled")
    reasoning_format: Literal["separated", "tagged"] = Field(default="tagged")

    @field_validator("prompt_config", mode="before")
    @classmethod
    def convert_none_prompt_config(cls, v: Any):
        if v is None:
            return PromptConfig()
        return v

    @property
    def structured_output_enabled(self) -> bool:
        """Backward compatibility property"""
        return self.structured_output_switch_on and self.structured_output is not None


class CodeNodeData(BaseNodeData):
    """Code node - execute Python/JavaScript code"""
    type: Literal["code"] = Field(default="code")

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

    # Removed type field to match dify/api
    variables: List[VariableSelector]
    code_language: Literal[CodeLanguage.PYTHON3, CodeLanguage.JAVASCRIPT]
    code: str
    outputs: Dict[str, Output]
    dependencies: Optional[List[Dependency]] = None


class HTTPRequestNodeData(BaseNodeData):
    """HTTP request node"""
    type: Literal["http-request"] = Field(default="http-request")

    class Authorization(BaseModel):
        """HTTP authorization configuration"""
        type: Literal["no-auth", "api-key", "bearer-token"] = "no-auth"
        config: Optional[Dict[str, str]] = None

    class Timeout(BaseModel):
        """HTTP timeout configuration matching Dify's HttpRequestNodeTimeout"""
        connect: int = 10
        read: int = 30
        write: int = 30

    # Align with Dify: headers and params are strings, not dicts
    method: str = "GET"
    url: str = Field(min_length=1, description="Request URL")
    headers: str = ""
    params: str = ""
    body: Optional[Dict[str, Any]] = None
    authorization: Authorization = Field(default_factory=Authorization)
    timeout: Optional[Timeout] = None

    @field_validator('method', mode='before')
    @classmethod
    def normalize_method(cls, v):
        if v is None:
            return "GET"
        if isinstance(v, str):
            v2 = v.strip().upper()
            allowed = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"}
            return v2 if v2 in allowed else v2
        return v

    @field_validator('headers', 'params', mode='before')
    @classmethod
    def coerce_string_fields(cls, v):
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            import json
            return json.dumps(v)
        if isinstance(v, list):
            import json
            return json.dumps(v)
        return str(v)

    @field_validator('timeout', mode='before')
    @classmethod
    def coerce_timeout(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            return cls.Timeout(**v)
        if isinstance(v, (int, float)):
            # Convert single timeout value to timeout object
            timeout_val = int(v)
            return cls.Timeout(connect=timeout_val, read=timeout_val, write=timeout_val)
        if isinstance(v, str):
            try:
                timeout_val = int(float(v.strip()))
                return cls.Timeout(connect=timeout_val, read=timeout_val, write=timeout_val)
            except (ValueError, TypeError):
                return None
        return v

    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://', '{{')):
            raise ValueError('URL must be a valid HTTP(S) URL or template variable')
        return v

    model_config = ConfigDict(extra="allow")


class ToolNodeData(BaseNodeData):
    """Tool node - external tool integration"""
    type: Literal["tool"] = Field(default="tool")

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
    type: Literal["if-else"] = Field(default="if-else")

    class Condition(BaseModel):
        """Single condition"""
        variable_selector: List[str] = Field(min_items=1)
        comparison_operator: Literal[
            "=", "≠", ">", "<", "≥", "≤", 
            "contains", "not contains", "start with", "end with", "starts with", "ends with",
            "is", "is not", "empty", "not empty", "in", "not in", "all of",
            "null", "not null", "exists", "not exists"
        ]
        value: str

    class Case(BaseModel):
        """Condition case"""
        case_id: str = Field(min_length=1)
        conditions: List["IfElseNodeData.Condition"] = Field(min_items=1)
        logical_operator: Literal["and", "or"] = "and"

    # Removed type field to match dify/api
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
    type: Literal["template-transform"] = Field(default="template-transform")
    template: str = Field(min_length=1, description="Jinja2 template")

    @field_validator('template')
    @classmethod
    def validate_template(cls, v):
        if not v.strip():
            raise ValueError('template cannot be empty')
        return v


class VariableAssignerNodeData(BaseNodeData):
    """Variable assigner node"""
    type: Literal["assigner", "variable-assigner"] = Field(default="assigner")
    assigned_variable_selector: Optional[List[str]] = None
    input_variable_selector: Optional[List[str]] = None
    write_mode: Literal["over-write", "append", "clear"] = "over-write"

    @model_validator(mode="before")
    @classmethod
    def accept_alt_keys(cls, data: Any):
        if isinstance(data, dict):
            # Accept alternative field names used in some DSLs
            if 'assigned_variable_selector' not in data and 'assigned_variable' in data:
                v = data.get('assigned_variable')
                data['assigned_variable_selector'] = v if isinstance(v, list) else [str(v)] if v is not None else None
            if 'input_variable_selector' not in data and 'input_variable' in data:
                v = data.get('input_variable')
                data['input_variable_selector'] = v if isinstance(v, list) else [str(v)] if v is not None else None
        return data

    model_config = ConfigDict(extra="allow")


class KnowledgeRetrievalNodeData(BaseNodeData):
    """Knowledge retrieval node"""
    type: Literal["knowledge-retrieval"] = Field(default="knowledge-retrieval")
    dataset_ids: List[str] = Field(min_items=1)
    query_variable_selector: List[str] = Field(min_items=1)
    retrieval_mode: Literal["single", "multiple"] = "single"
    top_k: int = Field(default=3, ge=1, le=20)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentNodeData(BaseNodeData):
    """Agent node - intelligent agent interaction"""
    type: Literal["agent"] = Field(default="agent")
    # Make strategy optional to align with Dify DSL variants where strategy metadata is split
    agent_strategy: Optional[str] = None
    agent_parameters: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class IterationNodeData(BaseNodeData):
    """Iteration node - loop processing"""
    type: Literal["iteration"] = Field(default="iteration")
    iterator_selector: List[str] = Field(min_items=1)
    output_selector: List[str] = Field(min_items=1)
    output_type: SegmentType = SegmentType.ARRAY_OBJECT


class ParameterExtractorNodeData(BaseNodeData):
    """Parameter extractor node"""
    type: Literal["parameter-extractor"] = Field(default="parameter-extractor")
    query: Union[str, List[str]] = Field(description="Query string or list of query strings")
    parameters: List[Dict[str, Any]] = Field(min_items=1)
    model: Optional[ModelConfig] = None
    instruction: Optional[str] = None
    reasoning_mode: Literal["function_call", "prompt"] = "function_call"
    vision: Optional[VisionConfig] = None

    @field_validator('query', mode='before')
    @classmethod
    def normalize_query(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


class QuestionClassifierNodeData(BaseNodeData):
    """Question classifier node"""
    type: Literal["question-classifier"] = Field(default="question-classifier")
    query_variable_selector: List[str] = Field(min_items=1)
    classes: List[Dict[str, Any]] = Field(min_items=2)


class IterationStartNodeData(BaseNodeData):
    """Iteration start pseudo node used by Dify to mark loop entry"""
    type: Literal["iteration-start"] = Field(default="iteration-start")
    model_config = ConfigDict(extra="allow")

    @field_validator('title', mode='before')
    @classmethod
    def ensure_title(cls, v):
        return v if v and v.strip() else "Iteration Start"


class LoopStartNodeData(BaseNodeData):
    """Loop start pseudo node"""
    type: Literal["loop-start"] = Field(default="loop-start")
    model_config = ConfigDict(extra="allow")


class LoopEndNodeData(BaseNodeData):
    """Loop end pseudo node"""
    type: Literal["loop-end"] = Field(default="loop-end")
    model_config = ConfigDict(extra="allow")


class VariableAggregatorNodeData(BaseNodeData):
    """Variable aggregator node - aggregate values across iterations or lists"""
    type: Literal["variable-aggregator"] = Field(default="variable-aggregator")
    # Keep permissive; different DSLs use various shapes
    aggregator: Optional[str] = None
    source_selector: Optional[List[str]] = None
    model_config = ConfigDict(extra="allow")


class DocumentExtractorNodeData(BaseNodeData):
    """Document extractor node - extract text from documents"""
    type: Literal["document-extractor"] = Field(default="document-extractor")
    variable_selector: List[str] = Field(min_items=1, description="Input variable selector")
    model_config = ConfigDict(extra="allow")


class ListOperatorNodeData(BaseNodeData):
    """List operator node - perform operations on lists"""
    type: Literal["list-operator"] = Field(default="list-operator")

    class FilterCondition(BaseModel):
        """Filter condition for list operations"""
        variable_selector: List[str] = Field(min_items=1)
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

    @field_validator('variable_selector', mode='before')
    @classmethod
    def ensure_variable_selector(cls, v):
        if v is None or v == []:
            return ["sys", "query"]
        return v


class NoteNodeData(BaseModel):
    """Note node - annotation/comment node for workflow documentation"""
    type: Literal[""] = Field(default="")
    title: str = Field(default="")
    theme: Literal["blue", "cyan", "green", "yellow", "pink", "violet"] = "blue"
    text: Optional[str] = None
    author: str = Field(default="")
    showAuthor: bool = Field(default=False)
    width: int = Field(default=240)
    height: int = Field(default=88)
    
    # Note nodes can have complex rich text data structure
    data: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(extra="allow")
    
    @field_validator('title', mode='before')
    @classmethod
    def normalize_title(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return "Note"
        return str(v).strip()





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
