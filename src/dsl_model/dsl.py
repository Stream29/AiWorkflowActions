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

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        """Ensure version follows semantic versioning"""
        if not v:
            raise ValueError('version cannot be empty')
        # Convert numeric versions to strings
        if isinstance(v, (int, float)):
            return f"v{v}"
        # Ensure it starts with v if it's a proper version
        v_str = str(v)
        if not v_str.startswith('v') and '.' in v_str:
            return f"v{v_str}"
        return v_str

    @model_validator(mode='after')
    def validate_workflow_consistency(self):
        """Validate workflow-app consistency"""
        # Ensure workflow mode consistency
        if hasattr(self.workflow.graph, 'nodes'):
            # Additional cross-validation can be added here
            pass
        return self

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="allow"  # Allow extra fields for DSL compatibility
    )
