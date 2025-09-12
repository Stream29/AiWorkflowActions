import json
from abc import ABC
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .enums import VariableType, LLMMode, ErrorStrategy, DefaultValueType


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


class PromptMessage(BaseModel):
    """LLM prompt message"""
    role: Literal["system", "user", "assistant"]
    text: str = Field(default="")
    id: Optional[str] = None


class PromptConfig(BaseModel):
    """Prompt configuration for Jinja2 variables"""
    jinja2_variables: Sequence[VariableSelector] = Field(default_factory=list)


class LLMNodeChatModelMessage(PromptMessage):
    """Enhanced chat model message for LLM nodes"""
    jinja2_text: Optional[str] = None


class LLMNodeCompletionModelPromptTemplate(BaseModel):
    """Completion model prompt template for LLM nodes"""
    jinja2_text: Optional[str] = None


class ContextConfig(BaseModel):
    """Context configuration for LLM nodes"""
    enabled: bool = Field(default=False)
    variable_selector: Optional[List[str]] = None

    @model_validator(mode='after')
    def validate_context_config(self):
        if self.enabled and not self.variable_selector:
            raise ValueError('variable_selector is required when context is enabled')
        return self


class VisionConfigOptions(BaseModel):
    """Vision configuration options"""
    variable_selector: Sequence[str] = Field(default_factory=lambda: ["sys", "files"])
    detail: Literal["low", "high"] = Field(default="high")


class VisionConfig(BaseModel):
    """Vision configuration for LLM nodes"""
    enabled: bool = Field(default=False)
    configs: VisionConfigOptions = Field(default_factory=VisionConfigOptions)


NumberType = Union[int, float]


class DefaultValue(BaseModel):
    """Default value configuration for error handling"""
    value: Any = None
    type: DefaultValueType
    key: str


class RetryConfig(BaseModel):
    """Node retry configuration"""
    max_retries: int = 0
    retry_interval: int = 0
    retry_enabled: bool = False

    @property
    def retry_interval_seconds(self) -> float:
        return self.retry_interval / 1000


class BaseNodeData(BaseModel, ABC):
    """Abstract base class for all node data types"""

    title: str = Field(description="Node display title")
    desc: Optional[str] = Field(default=None, description="Node description")
    version: str = Field(default="1", description="Node version")
    error_strategy: Optional[ErrorStrategy] = Field(default=None)
    default_value: Optional[List[DefaultValue]] = Field(default=None)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
