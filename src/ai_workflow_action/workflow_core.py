"""
Core workflow manipulation functionality
Handles loading, saving, and modifying workflow DSL files
"""

import yaml
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class WorkflowCore:
    """Core workflow operations without validation or AI logic"""
    
    def __init__(self):
        self.workflow_data: Dict[str, Any] = {}
        self.file_path: Optional[str] = None
    
    def load(self, file_path: str) -> None:
        """Load workflow from YAML file"""
        self.file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            self.workflow_data = yaml.safe_load(f)
    
    def save(self, file_path: Optional[str] = None) -> str:
        """Save workflow to YAML file"""
        output_path = file_path or self.file_path
        if not output_path:
            raise ValueError("No file path specified")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.workflow_data, f, 
                     default_flow_style=False,
                     allow_unicode=True,
                     sort_keys=False)
        
        return output_path
    
    def add_node_at_end(self, new_node: Dict[str, Any]) -> str:
        """Add a new node at the end of the workflow"""
        # Find terminal nodes (nodes with no outgoing edges)
        terminal_nodes = []
        for node in self.nodes:
            node_id = node.get('id')
            connections = self.get_node_connections(node_id)
            if not connections['outgoing']:
                terminal_nodes.append(node)
        
        if not terminal_nodes:
            raise ValueError("No terminal nodes found in workflow")
        
        # Use the first terminal node as reference
        terminal_node = terminal_nodes[0]
        return self.add_node_after(terminal_node['id'], new_node)
    
    def get_terminal_nodes(self) -> List[Dict[str, Any]]:
        """Get all terminal nodes (nodes with no outgoing edges)"""
        terminal_nodes = []
        for node in self.nodes:
            node_id = node.get('id')
            connections = self.get_node_connections(node_id)
            if not connections['outgoing']:
                terminal_nodes.append(node)
        return terminal_nodes
    
    @property
    def nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the workflow"""
        return self.workflow_data.get('workflow', {}).get('graph', {}).get('nodes', [])
    
    @property
    def edges(self) -> List[Dict[str, Any]]:
        """Get all edges in the workflow"""
        return self.workflow_data.get('workflow', {}).get('graph', {}).get('edges', [])
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Find a node by ID"""
        for node in self.nodes:
            if node.get('id') == node_id:
                return node
        return None
    
    def get_node_connections(self, node_id: str) -> Dict[str, List[str]]:
        """Get incoming and outgoing connections for a node"""
        incoming = []
        outgoing = []
        
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
        
        # Add node to workflow
        self.workflow_data['workflow']['graph']['nodes'].append(new_node)
        
        # Update edges
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
        
        # Remove the node
        self.workflow_data['workflow']['graph']['nodes'] = [
            n for n in self.nodes if n.get('id') != node_id
        ]
        
        # Remove edges connected to this node
        self.workflow_data['workflow']['graph']['edges'] = [
            e for e in self.edges 
            if e.get('source') != node_id and e.get('target') != node_id
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
        
        return True
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get summary information about the workflow"""
        node_types = {}
        for node in self.nodes:
            node_type = node.get('data', {}).get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
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
        
        # Try timestamp + random approach first (Dify style)
        for _ in range(100):  # Max attempts
            timestamp = int(time.time() * 1000)  # Milliseconds
            random_suffix = random.randint(100, 999)
            node_id = f"{timestamp}{random_suffix}"
            
            if node_id not in existing_ids:
                return node_id
        
        # Fallback to simple counter-based IDs
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
        
        # Check if there's a node after this one
        connections = self.get_node_connections(after_node_id)
        if connections['outgoing']:
            next_node = self.get_node(connections['outgoing'][0])
            if next_node and 'position' in next_node:
                next_pos = next_node['position']
                # Place new node between them
                return {
                    'x': (ref_pos['x'] + next_pos['x']) / 2,
                    'y': (ref_pos['y'] + next_pos['y']) / 2
                }
        
        # Default: place to the right
        return {'x': ref_pos['x'] + 200, 'y': ref_pos['y']}
    
    def _insert_node_in_edges(self, after_id: str, new_id: str,
                             after_type: str, new_type: str) -> None:
        """Insert new node in the edge chain"""
        # Find outgoing edge from after_node
        outgoing_edge = None
        for edge in self.edges:
            if edge.get('source') == after_id:
                outgoing_edge = edge
                break
        
        if outgoing_edge:
            target_id = outgoing_edge['target']
            target_node = self.get_node(target_id)
            target_type = target_node['data']['type'] if target_node else 'unknown'
            
            # Remove old edge
            self.workflow_data['workflow']['graph']['edges'].remove(outgoing_edge)
            
            # Add edge from after_node to new_node
            self._add_edge(after_id, new_id, after_type, new_type)
            
            # Add edge from new_node to original target
            self._add_edge(new_id, target_id, new_type, target_type)
        else:
            # No outgoing edge, just connect to new node
            self._add_edge(after_id, new_id, after_type, new_type)
    
    def _add_edge(self, source_id: str, target_id: str,
                  source_type: str, target_type: str) -> None:
        """Add an edge between two nodes"""
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