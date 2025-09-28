import random
import time
from typing import Dict, List, Optional, Set

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
                data=self.dsl.model_dump(mode='json', exclude_unset=True),
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

    def remove_node(self, node_id: str) -> None:
        """
        Remove a single node and all edges connected to it

        Args:
            node_id: ID of the node to remove

        Raises:
            ValueError: If node with given ID is not found
        """
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node '{node_id}' not found")

        # Remove the node
        self.dsl.workflow.graph.nodes = [
            n for n in self.dsl.workflow.graph.nodes if n.id != node_id
        ]

        # Remove all edges connected to this node
        self.dsl.workflow.graph.edges = [
            e for e in self.dsl.workflow.graph.edges
            if e.source != node_id and e.target != node_id
        ]

    def remove_nodes_after(self, node_id: str) -> List[str]:
        """
        Remove all nodes that come after the specified node
        Handles iteration/loop nodes specially

        Args:
            node_id: ID of the node whose successors to remove

        Returns:
            List of removed node IDs
        """
        node = self.get_node(node_id)
        if not node:
            return []

        # Find all nodes to remove using BFS
        nodes_to_remove: Set[str] = set()
        queue: List[str] = []

        # Get direct successors
        connections = self.get_node_connections(node_id)
        queue.extend(connections.outgoing)

        # Special handling for iteration/loop nodes
        if node.data.type in ['iteration', 'iteration-start', 'loop-start']:
            # If the node is an iteration/loop node, we need to find its end node
            # and remove everything after the end node as well
            nodes_to_remove.update(self._get_iteration_or_loop_contents(node_id))
            # Also add nodes after the iteration/loop end
            for n_id in list(nodes_to_remove):
                conn = self.get_node_connections(n_id)
                queue.extend(conn.outgoing)

        # BFS to find all downstream nodes
        while queue:
            current = queue.pop(0)
            if current not in nodes_to_remove:
                nodes_to_remove.add(current)
                connections = self.get_node_connections(current)
                queue.extend(connections.outgoing)

                # If we encounter another iteration/loop node, include its contents
                current_node = self.get_node(current)
                if current_node and current_node.data.type in ['iteration', 'iteration-start', 'loop-start']:
                    nodes_to_remove.update(self._get_iteration_or_loop_contents(current))

        # Remove all identified nodes and their edges
        removed_nodes: List[str] = []
        for node_to_remove in nodes_to_remove:
            try:
                self.remove_node(node_to_remove)
                removed_nodes.append(node_to_remove)
            except ValueError:
                # Node may have already been removed as part of iteration/loop cleanup
                pass

        return removed_nodes

    def _get_iteration_or_loop_contents(self, start_node_id: str) -> Set[str]:
        """
        Get all nodes within an iteration or loop block

        Args:
            start_node_id: ID of the iteration/loop start node

        Returns:
            Set of node IDs within the iteration/loop
        """
        start_node = self.get_node(start_node_id)
        if not start_node:
            return set()

        # For iteration/loop nodes, we need to find all nodes marked as being inside
        nodes_in_block: Set[str] = set()

        # Check edges for isInIteration or isInLoop flags
        for edge in self.dsl.workflow.graph.edges:
            if edge.data.isInIteration or edge.data.isInLoop:
                # Add both source and target if they're part of this iteration/loop
                # This is a simplified approach - in reality, we'd need to trace
                # the specific iteration/loop block
                nodes_in_block.add(edge.source)
                nodes_in_block.add(edge.target)

        # Alternative approach: find nodes between start and end nodes
        if start_node.data.type in ['loop-start']:
            # Find corresponding loop-end node
            for node in self.dsl.workflow.graph.nodes:
                if node.data.type == 'loop-end':
                    # Assume nodes are paired (simplified - real implementation would track pairing)
                    nodes_in_block.add(node.id)
                    break

        return nodes_in_block

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
