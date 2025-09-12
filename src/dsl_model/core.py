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


class PromptConfig(BaseModel):
    """Prompt configuration for Jinja2 variables"""
    jinja2_variables: Sequence[VariableSelector] = Field(default_factory=list)

    @field_validator("jinja2_variables", mode="before")
    @classmethod
    def convert_none_jinja2_variables(cls, v: Any):
        if v is None:
            return []
        return v


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

    @field_validator("configs", mode="before")
    @classmethod
    def convert_none_configs(cls, v: Any):
        if v is None:
            return VisionConfigOptions()
        return v


NumberType = Union[int, float]


class DefaultValue(BaseModel):
    """Default value configuration for error handling"""
    value: Any = None
    type: DefaultValueType
    key: str

    @staticmethod
    def _parse_json(value: str):
        """Unified JSON parsing handler"""
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format for value: {value}")

    @staticmethod
    def _validate_array(value: Any, element_type: type) -> bool:
        """Unified array type validation"""
        return isinstance(value, list) and all(isinstance(x, element_type) for x in value)

    @staticmethod
    def _convert_number(value: str) -> float:
        """Unified number conversion handler"""
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot convert to number: {value}")

    @model_validator(mode="after")
    def validate_value_type(self) -> "DefaultValue":
        if self.type is None:
            raise ValueError("type field is required")

        # Type validation configuration
        type_validators = {
            DefaultValueType.STRING: {
                "type": str,
                "converter": lambda x: x,
            },
            DefaultValueType.NUMBER: {
                "type": NumberType,
                "converter": self._convert_number,
            },
            DefaultValueType.OBJECT: {
                "type": dict,
                "converter": self._parse_json,
            },
            DefaultValueType.ARRAY_NUMBER: {
                "type": list,
                "element_type": NumberType,
                "converter": self._parse_json,
            },
            DefaultValueType.ARRAY_STRING: {
                "type": list,
                "element_type": str,
                "converter": self._parse_json,
            },
            DefaultValueType.ARRAY_OBJECT: {
                "type": list,
                "element_type": dict,
                "converter": self._parse_json,
            },
        }

        validator: dict[str, Any] = type_validators.get(self.type, {})
        if not validator:
            if self.type == DefaultValueType.ARRAY_FILES:
                # Handle files type
                return self
            raise ValueError(f"Unsupported type: {self.type}")

        # Handle string input cases
        if isinstance(self.value, str) and self.type != DefaultValueType.STRING:
            self.value = validator["converter"](self.value)

        # Validate base type
        if not isinstance(self.value, validator["type"]):
            raise ValueError(f"Value must be {validator['type'].__name__} type for {self.value}")

        # Validate array element types
        if validator["type"] == list and not self._validate_array(self.value, validator["element_type"]):
            raise ValueError(f"All elements must be {validator['element_type'].__name__} for {self.value}")

        return self


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

    title: str = Field(min_length=1, description="Node display title")
    desc: Optional[str] = Field(default=None, description="Node description")
    version: str = Field(default="1", description="Node version")
    error_strategy: Optional[ErrorStrategy] = Field(default=None)
    default_value: Optional[List[DefaultValue]] = Field(default=None)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    @property
    def default_value_dict(self) -> Dict[str, Any]:
        """Convert default_value list to dictionary for easy access"""
        if self.default_value:
            return {item.key: item.value for item in self.default_value}
        return {}

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if not v.strip():
            raise ValueError('title cannot be empty')
        return v.strip()
