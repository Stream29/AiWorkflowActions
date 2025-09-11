from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .enums import NodeType, SegmentType, CodeLanguage
from .core import (
    BaseNodeData,
    ContextConfig,
    ModelConfig,
    PromptMessage,
    Variable,
    VariableSelector,
    VisionConfig,
)


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
