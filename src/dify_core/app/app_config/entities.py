from collections.abc import Sequence
from enum import Enum, StrEnum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModelConfigEntity(BaseModel):
    """
    Model Config Entity.
    """

    provider: str
    model: str
    mode: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    stop: list[str] = Field(default_factory=list)


class VariableEntityType(StrEnum):
    TEXT_INPUT = "text-input"
    SELECT = "select"
    PARAGRAPH = "paragraph"
    NUMBER = "number"
    EXTERNAL_DATA_TOOL = "external_data_tool"
    FILE = "file"
    FILE_LIST = "file-list"
    CHECKBOX = "checkbox"


class VariableEntity(BaseModel):
    """
    Variable Entity.
    """

    # `variable` records the name of the variable in user inputs.
    variable: str
    label: str
    description: str = ""
    type: VariableEntityType
    required: bool = False
    hide: bool = False
    max_length: Optional[int] = None
    options: Sequence[str] = Field(default_factory=list)
    # Simplified file handling fields
    allowed_file_types: Sequence[str] = Field(default_factory=list)
    allowed_file_extensions: Sequence[str] = Field(default_factory=list)
    allowed_file_upload_methods: Sequence[str] = Field(default_factory=list)

    @field_validator("description", mode="before")
    @classmethod
    def convert_none_description(cls, v: Any) -> str:
        return v or ""

    @field_validator("options", mode="before")
    @classmethod
    def convert_none_options(cls, v: Any) -> Sequence[str]:
        return v or []


SupportedComparisonOperator = Literal[
    # for string or array
    "contains",
    "not contains",
    "start with",
    "end with",
    "is",
    "is not",
    "empty",
    "not empty",
    "in",
    "not in",
    # for number
    "=",
    "≠",
    ">",
    "<",
    "≥",
    "≤",
    # for time
    "before",
    "after",
]


class ModelConfig(BaseModel):
    provider: str
    name: str
    mode: str
    completion_params: dict[str, Any] = Field(default_factory=dict)


class Condition(BaseModel):
    """
    Condition detail
    """

    name: str
    comparison_operator: SupportedComparisonOperator
    value: str | Sequence[str] | None | int | float = None