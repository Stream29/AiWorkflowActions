"""
Core workflow manipulation functionality
Handles loading, saving, and modifying workflow DSL files
"""

import yaml
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from dsl_model import DifyWorkflowDSL, Node, Edge, EdgeData, Position


class WorkflowCore:
    """Core workflow operations without validation or AI logic, now backed by dsl_model Pydantic models"""
    
    def __init__(self):
        self.workflow_data: Dict[str, Any] = {}
        self.file_path: Optional[str] = None
        self.dsl: Optional[DifyWorkflowDSL] = None
    
    def _sync_dict_from_model(self) -> None:
        """Keep legacy dict view in sync with Pydantic model"""
        if self.dsl is not None:
            # Use python mode to keep enums as values (dsl config use_enum_values=True)
            self.workflow_data = self.dsl.model_dump()
    
    def load(self, file_path: str) -> None:
        """Load workflow from YAML file and parse into DSL model"""
        self.file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # Parse and validate using DSL model
        self.dsl = DifyWorkflowDSL.model_validate(data)
        self._sync_dict_from_model()
    
    def save(self, file_path: Optional[str] = None) -> str:
        """Save workflow to YAML file (serialized from DSL model when available)"""
        output_path = file_path or self.file_path
        if not output_path:
            raise ValueError("No file path specified")
        
        # Ensure dict is in sync with model
        if self.dsl is not None:
            self._sync_dict_from_model()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.workflow_data, f, 
                     default_flow_style=False,
                     allow_unicode=True,
                     sort_keys=False)
        
        return output_path
    
    def add_node_at_end(self, new_node: Dict[str, Any]) -> str:
        """Add a new node at the end of the workflow"""
        terminal_nodes = self.get_terminal_nodes()
        if not terminal_nodes:
            raise ValueError("No terminal nodes found in workflow")
        terminal_node = terminal_nodes[0]
        return self.add_node_after(terminal_node['id'], new_node)
    
    def get_terminal_nodes(self) -> List[Dict[str, Any]]:
        """Get all terminal nodes (nodes with no outgoing edges)"""
        terminal_nodes: List[Dict[str, Any]] = []
        for node in self.nodes:
            node_id = node.get('id')
            connections = self.get_node_connections(node_id)
            if not connections['outgoing']:
                terminal_nodes.append(node)
        return terminal_nodes
    
    @property
    def nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the workflow (as dicts for backward compatibility)"""
        if self.dsl is not None:
            return [n.model_dump() for n in self.dsl.workflow.graph.nodes]
        return self.workflow_data.get('workflow', {}).get('graph', {}).get('nodes', [])
    
    @property
    def edges(self) -> List[Dict[str, Any]]:
        """Get all edges in the workflow (as dicts for backward compatibility)"""
        if self.dsl is not None:
            return [e.model_dump() for e in self.dsl.workflow.graph.edges]
        return self.workflow_data.get('workflow', {}).get('graph', {}).get('edges', [])
    
    def _get_node_model(self, node_id: str) -> Optional[Node]:
        if self.dsl is None:
            return None
        for n in self.dsl.workflow.graph.nodes:
            if n.id == node_id:
                return n
        return None
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Find a node by ID (dict view)"""
        if self.dsl is not None:
            n = self._get_node_model(node_id)
            return n.model_dump() if n else None
        for node in self.nodes:
            if node.get('id') == node_id:
                return node
        return None
    
    def get_node_connections(self, node_id: str) -> Dict[str, List[str]]:
        """Get incoming and outgoing connections for a node"""
        incoming: List[str] = []
        outgoing: List[str] = []
        if self.dsl is not None:
            for edge in self.dsl.workflow.graph.edges:
                if edge.source == node_id:
                    outgoing.append(edge.target)
                if edge.target == node_id:
                    incoming.append(edge.source)
            return {'incoming': incoming, 'outgoing': outgoing}
        
        for edge in self.edges:
            if edge.get('source') == node_id:
                outgoing.append(edge.get('target'))
            if edge.get('target') == node_id:
                incoming.append(edge.get('source'))
        
        return {'incoming': incoming, 'outgoing': outgoing}
    
    def add_node_after(self, after_node_id: str, new_node: Dict[str, Any]) -> str:
        """
        Add a new node after specified node
        Returns the new node's ID
        """
        # Verify after_node exists
        after_node = self.get_node(after_node_id)
        if not after_node:
            raise ValueError(f"Node '{after_node_id}' not found")
        
        # Generate unique node ID if not provided
        node_id = new_node.get('id')
        if not node_id:
            node_id = self._generate_node_id(new_node['data']['type'])
            new_node['id'] = node_id
        
        # Set position if not provided
        if 'position' not in new_node:
            new_node['position'] = self._calculate_position(after_node_id)
        # Ensure positionAbsolute for model requirements
        if 'positionAbsolute' not in new_node:
            new_node['positionAbsolute'] = {
                'x': new_node['position']['x'],
                'y': new_node['position']['y'],
            }
        
        if self.dsl is not None:
            # Validate/construct Node via Pydantic and add to model graph
            node_model = Node.model_validate(new_node)
            self.dsl.workflow.graph.nodes.append(node_model)
            # Update edges in model
            self._insert_node_in_edges(after_node_id, node_id, 
                                       after_node['data']['type'],
                                       new_node['data']['type'])
            # Sync dict view
            self._sync_dict_from_model()
            return node_id
        
        # Fallback (no model): Add node to workflow dict
        self.workflow_data['workflow']['graph']['nodes'].append(new_node)
        self._insert_node_in_edges(after_node_id, node_id, 
                                   after_node['data']['type'],
                                   new_node['data']['type'])
        return node_id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and reconnect edges"""
        node = self.get_node(node_id)
        if not node:
            return False
        
        # Get connections
        connections = self.get_node_connections(node_id)
        
        if self.dsl is not None:
            # Remove node from model
            self.dsl.workflow.graph.nodes = [n for n in self.dsl.workflow.graph.nodes if n.id != node_id]
            # Remove connected edges
            self.dsl.workflow.graph.edges = [
                e for e in self.dsl.workflow.graph.edges
                if e.source != node_id and e.target != node_id
            ]
            # Reconnect edges (bridge the gap)
            for source in connections['incoming']:
                for target in connections['outgoing']:
                    source_node = self.get_node(source)
                    target_node = self.get_node(target)
                    if source_node and target_node:
                        self._add_edge(source, target,
                                       source_node['data']['type'],
                                       target_node['data']['type'])
            self._sync_dict_from_model()
            return True
        
        # Fallback to dict manipulation
        self.workflow_data['workflow']['graph']['nodes'] = [
            n for n in self.nodes if n.get('id') != node_id
        ]
        self.workflow_data['workflow']['graph']['edges'] = [
            e for e in self.edges 
            if e.get('source') != node_id and e.get('target') != node_id
        ]
        for source in connections['incoming']:
            for target in connections['outgoing']:
                source_node = self.get_node(source)
                target_node = self.get_node(target)
                if source_node and target_node:
                    self._add_edge(source, target,
                                   source_node['data']['type'],
                                   target_node['data']['type'])
        return True
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get summary information about the workflow"""
        node_types: Dict[str, int] = {}
        for node in self.nodes:
            node_type = node.get('data', {}).get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        if self.dsl is not None:
            app = self.dsl.app
            return {
                'app_name': app.name,
                'description': app.description,
                'mode': app.mode,
                'node_count': len(self.dsl.workflow.graph.nodes),
                'edge_count': len(self.dsl.workflow.graph.edges),
                'node_types': node_types
            }
        
        return {
            'app_name': self.workflow_data.get('app', {}).get('name', 'Unknown'),
            'description': self.workflow_data.get('app', {}).get('description', ''),
            'mode': self.workflow_data.get('app', {}).get('mode', 'workflow'),
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'node_types': node_types
        }
    
    def _generate_node_id(self, node_type: str) -> str:
        """Generate a unique node ID in Dify format (timestamp-based)"""
        existing_ids = {node.get('id') for node in self.nodes}
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
    
    def _calculate_position(self, after_node_id: str) -> Dict[str, float]:
        """Calculate position for new node"""
        after_node = self.get_node(after_node_id)
        ref_pos = after_node.get('position', {'x': 100, 'y': 300})
        connections = self.get_node_connections(after_node_id)
        if connections['outgoing']:
            next_node = self.get_node(connections['outgoing'][0])
            if next_node and 'position' in next_node:
                next_pos = next_node['position']
                return {
                    'x': (ref_pos['x'] + next_pos['x']) / 2,
                    'y': (ref_pos['y'] + next_pos['y']) / 2
                }
        return {'x': ref_pos['x'] + 200, 'y': ref_pos['y']}
    
    def _insert_node_in_edges(self, after_id: str, new_id: str,
                             after_type: str, new_type: str) -> None:
        """Insert new node in the edge chain"""
        if self.dsl is not None:
            # Find outgoing edge from after_node in model
            outgoing_edge = None
            for edge in self.dsl.workflow.graph.edges:
                if edge.source == after_id:
                    outgoing_edge = edge
                    break
            if outgoing_edge:
                target_id = outgoing_edge.target
                # Remove old edge
                self.dsl.workflow.graph.edges.remove(outgoing_edge)
                # Add edge from after_node to new_node
                self._add_edge(after_id, new_id, after_type, new_type)
                # Add edge from new_node to original target
                target_node = self.get_node(target_id)
                target_type = target_node['data']['type'] if target_node else 'unknown'
                self._add_edge(new_id, target_id, new_type, target_type)
            else:
                self._add_edge(after_id, new_id, after_type, new_type)
            return
        
        # Fallback dict manipulation
        outgoing_edge = None
        for edge in self.edges:
            if edge.get('source') == after_id:
                outgoing_edge = edge
                break
        if outgoing_edge:
            target_id = outgoing_edge['target']
            target_node = self.get_node(target_id)
            target_type = target_node['data']['type'] if target_node else 'unknown'
            self.workflow_data['workflow']['graph']['edges'].remove(outgoing_edge)
            self._add_edge(after_id, new_id, after_type, new_type)
            self._add_edge(new_id, target_id, new_type, target_type)
        else:
            self._add_edge(after_id, new_id, after_type, new_type)
    
    def _add_edge(self, source_id: str, target_id: str,
                  source_type: str, target_type: str) -> None:
        """Add an edge between two nodes"""
        if self.dsl is not None:
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
            # Keep dict view in sync
            self._sync_dict_from_model()
            return
        
        edge = {
            'data': {
                'isInIteration': False,
                'sourceType': source_type,
                'targetType': target_type
            },
            'id': f'{source_id}-{target_id}',
            'source': source_id,
            'sourceHandle': 'source',
            'target': target_id,
            'targetHandle': 'target',
            'type': 'custom'
        }
        self.workflow_data['workflow']['graph']['edges'].append(edge)