import random
import time
from typing import Dict, List, Optional

import yaml

from dsl_model import DifyWorkflowDSL, Node, Edge, EdgeData, NodeData, Position
from .models import (
    NodeConnection,
    WorkflowInfo
)


class DifyWorkflowDslFile:
    """
    Second layer: DifyWorkflowDSL wrapper with basic operations and validation
    Implements RAII pattern - resource is acquired on construction, released on destruction
    """

    def __init__(self, file_path: str):
        """
        Initialize with optional file loading (RAII pattern)
        
        Args:
            file_path: Optional path to load workflow file immediately
        """
        self.file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        self.dsl = DifyWorkflowDSL.model_validate(data)

    # === File Operations ===

    def save(self, file_path: str):
        """Save workflow to YAML file (serialized from DSL model)"""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                data=self.dsl.model_dump(mode='json'),
                stream=f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )

    # === Workflow Information ===

    def get_workflow_info(self) -> WorkflowInfo:
        """Get summary information about the workflow"""
        node_types: Dict[str, int] = {}
        for node in self.dsl.workflow.graph.nodes:
            node_type = node.data.type
            node_types[node_type] = node_types.get(node_type, 0) + 1

        app = self.dsl.app
        return WorkflowInfo(
            app_name=app.name,
            description=app.description,
            mode=app.mode,
            node_count=len(self.dsl.workflow.graph.nodes),
            edge_count=len(self.dsl.workflow.graph.edges),
            node_types=node_types
        )

    # === Node Operations ===

    def get_node(self, node_id: str) -> Optional[Node]:
        for n in self.dsl.workflow.graph.nodes:
            if n.id == node_id:
                return n
        return None

    def get_node_connections(self, node_id: str) -> NodeConnection:
        """Get incoming and outgoing connections for a node"""
        incoming: List[str] = []
        outgoing: List[str] = []
        for edge in self.dsl.workflow.graph.edges:
            if edge.source == node_id:
                outgoing.append(edge.target)
            if edge.target == node_id:
                incoming.append(edge.source)
        return NodeConnection(incoming=incoming, outgoing=outgoing)

    def get_terminal_nodes(self) -> List[Node]:
        """Get all terminal nodes (nodes with no outgoing edges)"""
        terminal_nodes: List[Node] = []
        for node in self.dsl.workflow.graph.nodes:
            connections = self.get_node_connections(node.id)
            if not connections.outgoing:
                terminal_nodes.append(node)
        return terminal_nodes

    # === Node Modification ===

    def add_node_after(self, after_node_id: str, new_node_data: NodeData) -> str:
        """
        Add a new node after specified node
        Returns the new node's ID
        """
        after_node = self.get_node(after_node_id)
        if not after_node:
            raise ValueError(f"Node '{after_node_id}' not found")
        node_id = self._generate_node_id(new_node_data.type)
        position = self._calculate_position(after_node_id)
        node_model = Node(
            id=node_id,
            data=new_node_data,
            position=position,
            positionAbsolute=position,
            selected=False,
            sourcePosition='right',
            targetPosition='left',
            type='custom'
        )

        self.dsl.workflow.graph.nodes.append(node_model)
        self._insert_node_in_edges(
            after_id=after_node_id,
            new_id=node_id,
            after_type=after_node.data.type,
            new_type=new_node_data.type
        )
        return node_id

    # === Private Helper Methods ===

    def _generate_node_id(self, node_type: str) -> str:
        """Generate a unique node ID in Dify format (timestamp-based)"""
        existing_ids = {node.id for node in self.dsl.workflow.graph.nodes}
        for _ in range(100):  # Max attempts
            timestamp = int(time.time() * 1000)  # Milliseconds
            random_suffix = random.randint(100, 999)
            node_id = f"{timestamp}{random_suffix}"
            if node_id not in existing_ids:
                return node_id
        counter = 1
        while True:
            node_id = f"{node_type}-{counter}"
            if node_id not in existing_ids:
                return node_id
            counter += 1

    def _calculate_position(self, after_node_id: str) -> Position:
        """Calculate position for new node"""
        after_node = self.get_node(after_node_id)
        if not after_node:
            raise ValueError(f"Node '{after_node_id}' not found")
        ref_pos = after_node.position
        connections = self.get_node_connections(after_node_id)
        if connections.outgoing:
            next_node = self.get_node(connections.outgoing[0])
            if next_node:
                next_pos = next_node.position
                return Position(
                    x=(ref_pos.x + next_pos.x) / 2,
                    y=(ref_pos.y + next_pos.y) / 2
                )
        return Position(x=ref_pos.x + 200, y=ref_pos.y)

    def _insert_node_in_edges(self, after_id: str, new_id: str,
                              after_type: str, new_type: str) -> None:
        """Insert new node in the edge chain"""
        # Find outgoing edge from after_node in model
        outgoing_edge = None
        for edge in self.dsl.workflow.graph.edges:
            if edge.source == after_id:
                outgoing_edge = edge
                break
        if outgoing_edge:
            target_id = outgoing_edge.target
            self.dsl.workflow.graph.edges.remove(outgoing_edge)
            self._add_edge(after_id, new_id, after_type, new_type)
            target_node = self.get_node(target_id)
            target_type = target_node.data.type if target_node else 'unknown'
            self._add_edge(new_id, target_id, new_type, target_type)
        else:
            self._add_edge(after_id, new_id, after_type, new_type)

    def _add_edge(self, source_id: str, target_id: str,
                  source_type: str, target_type: str) -> None:
        """Add an edge between two nodes"""
        if self.dsl is None:
            raise ValueError("No workflow loaded")

        edge_model = Edge(
            id=f"{source_id}-{target_id}",
            source=source_id,
            target=target_id,
            sourceHandle='source',
            targetHandle='target',
            type='custom',
            data=EdgeData(
                isInIteration=False,
                isInLoop=False,
                sourceType=source_type,
                targetType=target_type,
            ),
        )
        self.dsl.workflow.graph.edges.append(edge_model)
