from dify_core.workflow.nodes.base.entities import BaseNodeData
from dify_core.workflow.entities.variable_entities import VariableSelector


class TemplateTransformNodeData(BaseNodeData):
    """Template Transform Node Data."""
    variables: list[VariableSelector]
    template: str