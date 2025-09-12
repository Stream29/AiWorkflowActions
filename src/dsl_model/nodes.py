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

    # Removed type field to match dify/api
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
        comparison_operator: Literal["=", "≠", ">", "<", "≥", "≤", "contains", "starts with", "ends with"]
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
    assigned_variable_selector: List[str] = Field(min_items=1)
    input_variable_selector: List[str] = Field(min_items=1)
    write_mode: Literal["over-write", "append", "clear"] = "over-write"


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
    agent_strategy: str = Field(min_length=1)
    agent_parameters: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None


class IterationNodeData(BaseNodeData):
    """Iteration node - loop processing"""
    type: Literal["iteration"] = Field(default="iteration")
    iterator_selector: List[str] = Field(min_items=1)
    output_selector: List[str] = Field(min_items=1)
    output_type: SegmentType = SegmentType.ARRAY_OBJECT


class ParameterExtractorNodeData(BaseNodeData):
    """Parameter extractor node"""
    type: Literal["parameter-extractor"] = Field(default="parameter-extractor")
    query: str = Field(min_length=1)
    parameters: List[Dict[str, Any]] = Field(min_items=1)


class QuestionClassifierNodeData(BaseNodeData):
    """Question classifier node"""
    type: Literal["question-classifier"] = Field(default="question-classifier")
    query_variable_selector: List[str] = Field(min_items=1)
    classes: List[Dict[str, Any]] = Field(min_items=2)


# For DSL compatibility, use a flexible base type with model_rebuild
class FlexibleNodeData(BaseModel):
    """Flexible node data for DSL compatibility"""
    type: str = Field(description="Node type identifier")
    title: str = Field(default="", description="Node display title")
    desc: str = Field(default="", description="Node description")

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
        ParameterExtractorNodeData,
        QuestionClassifierNodeData,
    ],
    Field(discriminator='type')
]
