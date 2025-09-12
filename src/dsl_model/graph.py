from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .core import Position, Viewport
from .nodes import NodeData
from .enums import NodeType


class EdgeData(BaseModel):
    """Edge connection metadata"""
    sourceType: str = Field(min_length=1)
    targetType: str = Field(min_length=1)
    isInIteration: bool = Field(default=False)
    isInLoop: bool = Field(default=False)


class Edge(BaseModel):
    """Graph edge definition"""
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    sourceHandle: str = Field(default="source")
    targetHandle: str = Field(default="target")
    type: str = Field(default="custom")
    data: EdgeData
    zIndex: int = Field(default=0)

    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not v.strip():
            raise ValueError('edge id cannot be empty')
        return v


class Node(BaseModel):
    """Graph node definition"""
    id: str = Field(min_length=1)
    data: NodeData
    position: Position
    positionAbsolute: Position
    height: int = Field(default=89, ge=1)
    width: int = Field(default=244, ge=1)
    selected: bool = Field(default=False)
    sourcePosition: Literal["left", "right", "top", "bottom"] = Field(default="right")
    targetPosition: Literal["left", "right", "top", "bottom"] = Field(default="left")
    type: str = Field(default="custom")

    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not v.strip():
            raise ValueError('node id cannot be empty')
        return v

    @model_validator(mode='after')
    def validate_node_consistency(self):
        """Ensure node data type matches expected structure"""
        if hasattr(self.data, 'type'):
            # Additional validation can be added here
            pass
        return self


class Graph(BaseModel):
    """Complete workflow graph"""
    nodes: List[Node] = Field(min_items=1)
    edges: List[Edge] = Field(default_factory=list)
    viewport: Viewport = Field(default_factory=Viewport)

    @model_validator(mode='after')
    def validate_graph_structure(self):
        """Validate graph connectivity and structure"""
        node_ids = {node.id for node in self.nodes}

        # Validate edges reference valid nodes
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f'Edge source "{edge.source}" references non-existent node')
            if edge.target not in node_ids:
                raise ValueError(f'Edge target "{edge.target}" references non-existent node')

        # Ensure there's exactly one start node
        from .nodes import StartNodeData
        # Prefer semantic detection by model type or discriminant field
        start_node_ids = set()
        for node in self.nodes:
            try:
                if isinstance(node.data, StartNodeData):
                    start_node_ids.add(node.id)
                    continue
                node_type_val = getattr(node.data, "type", None)
                if node_type_val == NodeType.START or node_type_val == NodeType.START.value:
                    start_node_ids.add(node.id)
            except Exception:
                # Be resilient to unexpected shapes
                pass
        # Fallback: infer from edges metadata if none found
        if not start_node_ids:
            for edge in self.edges:
                if edge.data and edge.data.sourceType == NodeType.START.value:
                    start_node_ids.add(edge.source)
        if len(start_node_ids) != 1:
            raise ValueError('Graph must have exactly one start node')

        return self
