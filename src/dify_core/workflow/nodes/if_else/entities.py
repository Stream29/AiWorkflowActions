from typing import Literal, Optional
from pydantic import BaseModel, Field

from dify_core.workflow.nodes.base.entities import BaseNodeData
from dify_core.app.app_config.entities import Condition


class IfElseNodeData(BaseNodeData):
    """If Else Node Data."""
    
    class Case(BaseModel):
        """Case entity representing a single logical condition group"""
        case_id: str
        logical_operator: Literal["and", "or"]
        conditions: list[Condition]
    
    logical_operator: Optional[Literal["and", "or"]] = "and"
    conditions: Optional[list[Condition]] = Field(default=None, deprecated=True)
    cases: Optional[list[Case]] = None