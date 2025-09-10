from typing import Any, Literal, Optional, Sequence, Mapping
from pydantic import BaseModel, Field

from dify_core.workflow.nodes.base.entities import BaseNodeData
from dify_core.app.app_config.entities import ModelConfig


class PromptConfig(BaseModel):
    """LLM prompt configuration"""
    jinja2_variables: Optional[list[dict]] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    """Memory configuration for LLM"""
    enabled: bool = False
    window: dict = Field(default_factory=dict)


class ContextConfig(BaseModel):
    """Context configuration for LLM"""
    enabled: bool = False
    variable_selector: list[str] = Field(default_factory=list)


class VisionConfig(BaseModel):
    """Vision configuration for LLM"""
    enabled: bool = False
    configs: dict = Field(default_factory=dict)


class LLMNodeChatModelMessage(BaseModel):
    """Chat model message for LLM node"""
    role: str
    text: str


class LLMNodeCompletionModelPromptTemplate(BaseModel):
    """Completion model prompt template"""
    text: str


class LLMNodeData(BaseNodeData):
    model: ModelConfig
    prompt_template: Sequence[LLMNodeChatModelMessage] | LLMNodeCompletionModelPromptTemplate
    prompt_config: PromptConfig = Field(default_factory=PromptConfig)
    memory: Optional[MemoryConfig] = None
    context: ContextConfig = Field(default_factory=ContextConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    structured_output: Mapping[str, Any] | None = None
    structured_output_switch_on: bool = Field(False, alias="structured_output_enabled")
    reasoning_format: Literal["separated", "tagged"] = Field(default="tagged")