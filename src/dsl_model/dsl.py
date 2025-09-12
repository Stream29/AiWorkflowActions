from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from dsl_model.app_models import AppMetadata, Dependency
from dsl_model.features import Workflow


class DifyWorkflowDSL(BaseModel):
    """Complete Dify Workflow DSL with strict validation"""
    app: AppMetadata
    dependencies: List[Dependency] = Field(default_factory=list)
    kind: str = Field(default="app")
    version: str = Field(default="v1.0.0", description="Workflow version")
    workflow: Workflow

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="allow"  # Allow extra fields for DSL compatibility
    )
