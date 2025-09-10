from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, AfterValidator

from dify_core.workflow.nodes.base.entities import BaseNodeData
from dify_core.workflow.entities.variable_entities import VariableSelector


class CodeLanguage:
    PYTHON3 = "python3"
    JAVASCRIPT = "javascript"


class SegmentType:
    STRING = "string"
    NUMBER = "number"
    OBJECT = "object"
    ARRAY_STRING = "array[string]"
    ARRAY_NUMBER = "array[number]"
    ARRAY_OBJECT = "array[object]"


def _validate_type(value):
    """Simple type validator"""
    return value


class CodeNodeData(BaseNodeData):
    """Code Node Data."""
    
    class Output(BaseModel):
        type: Annotated[str, AfterValidator(_validate_type)]
        children: Optional[dict[str, "CodeNodeData.Output"]] = None
    
    class Dependency(BaseModel):
        name: str
        version: str
    
    variables: list[VariableSelector]
    code_language: Literal["python3", "javascript"]
    code: str
    outputs: dict[str, Output]
    dependencies: Optional[list[Dependency]] = None