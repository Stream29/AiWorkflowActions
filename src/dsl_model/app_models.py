from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from .enums import WorkflowMode


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
