from collections.abc import Sequence

from pydantic import Field

from dify_core.app.app_config.entities import VariableEntity
from dify_core.workflow.nodes.base.entities import BaseNodeData


class StartNodeData(BaseNodeData):
    """
    Start Node Data
    """

    variables: Sequence[VariableEntity] = Field(default_factory=list)