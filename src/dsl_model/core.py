from abc import ABC
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .enums import VariableType, LLMMode


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


class BaseNodeData(BaseModel, ABC):
    """Abstract base class for all node data types"""

    type: Any = Field(description="Node type identifier")
    title: str = Field(min_length=1, description="Node display title")
    desc: str = Field(default="", description="Node description")
    version: str = Field(default="1", description="Node version")
    error_strategy: Optional[Any] = Field(default=None)

    # Allow extra fields for DSL compatibility
    model_config = ConfigDict(extra="allow")

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('title cannot be empty')
        return v.strip()
