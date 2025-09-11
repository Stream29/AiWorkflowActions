from typing import Any, Dict

from pydantic import BaseModel

from .enums import NodeType
from .core import Position
from .nodes import (
    AgentNodeData,
    AnswerNodeData,
    CodeNodeData,
    EndNodeData,
    HTTPRequestNodeData,
    IfElseNodeData,
    IterationNodeData,
    KnowledgeRetrievalNodeData,
    LLMNodeData,
    NodeData,
    ParameterExtractorNodeData,
    QuestionClassifierNodeData,
    StartNodeData,
    TemplateTransformNodeData,
    ToolNodeData,
    VariableAssignerNodeData,
)
from .graph import Edge, EdgeData, Node
from .features import Workflow
from .app_models import AppMetadata, Dependency
from .dsl import DifyWorkflowDSL


def create_node(
    node_id: str,
    node_type: NodeType,
    title: str,
    position: Position,
    **kwargs
) -> Node:
    """Create a properly typed node with validation"""

    # Map node type to data class
    node_data_map = {
        NodeType.START: StartNodeData,
        NodeType.END: EndNodeData,
        NodeType.ANSWER: AnswerNodeData,
        NodeType.LLM: LLMNodeData,
        NodeType.CODE: CodeNodeData,
        NodeType.HTTP_REQUEST: HTTPRequestNodeData,
        NodeType.TOOL: ToolNodeData,
        NodeType.IF_ELSE: IfElseNodeData,
        NodeType.TEMPLATE_TRANSFORM: TemplateTransformNodeData,
        NodeType.VARIABLE_ASSIGNER: VariableAssignerNodeData,
        NodeType.KNOWLEDGE_RETRIEVAL: KnowledgeRetrievalNodeData,
        NodeType.AGENT: AgentNodeData,
        NodeType.ITERATION: IterationNodeData,
        NodeType.PARAMETER_EXTRACTOR: ParameterExtractorNodeData,
        NodeType.QUESTION_CLASSIFIER: QuestionClassifierNodeData,
    }

    data_class = node_data_map.get(node_type)
    if not data_class:
        raise ValueError(f"Unsupported node type: {node_type}")

    # Create node data with proper type and validation
    node_data = data_class(
        type=node_type,
        title=title,
        **kwargs
    )

    return Node(
        id=node_id,
        data=node_data,
        position=position,
        positionAbsolute=position
    )


def create_edge(
    edge_id: str,
    source_id: str,
    target_id: str,
    source_type: str,
    target_type: str
) -> Edge:
    """Create a validated edge"""
    return Edge(
        id=edge_id,
        source=source_id,
        target=target_id,
        data=EdgeData(
            sourceType=source_type,
            targetType=target_type
        )
    )


def validate_dsl(data: Dict[str, Any]) -> DifyWorkflowDSL:
    """Validate DSL data and return parsed workflow"""
    try:
        return DifyWorkflowDSL.model_validate(data)
    except Exception as e:
        raise ValueError(f"DSL validation failed: {e}")
