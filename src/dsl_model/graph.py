from typing import Any, Dict, List, Literal, Optional

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
    extent: Optional[str] = Field(default=None)
    parentId: Optional[str] = Field(default=None)
    zIndex: Optional[int] = Field(default=None)
    type: str = Field(default="custom")


class Graph(BaseModel):
    """Complete workflow graph"""
    nodes: List[Node] = Field()
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
        return self
