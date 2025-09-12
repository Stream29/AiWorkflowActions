"""
DifyWorkflowDslFile - Second layer abstraction
Encapsulates DifyWorkflowDSL with basic operations and validation
Implements RAII pattern for resource management
"""

import random
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

import yaml
from pydantic import ValidationError

from dsl_model import DifyWorkflowDSL, Node, Edge, EdgeData
from dsl_model.enums import NodeType
from dsl_model.nodes import (
    StartNodeData,
    LLMNodeData,
    CodeNodeData,
    TemplateTransformNodeData,
    IfElseNodeData,
    HTTPRequestNodeData,
    EndNodeData,
)

from .models import (
    NodeConnection, 
    WorkflowInfo, 
    NodeValidationResult, 
    WorkflowValidationResult, 
    LinearityCheck
)


class DifyWorkflowDslFile:
    """
    Second layer: DifyWorkflowDSL wrapper with basic operations and validation
    Implements RAII pattern - resource is acquired on construction, released on destruction
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize with optional file loading (RAII pattern)
        
        Args:
            file_path: Optional path to load workflow file immediately
        """
        self.file_path: Optional[str] = None
        self.dsl: Optional[DifyWorkflowDSL] = None
        
        # Node validation models mapping
        self._node_models = {
            NodeType.START.value: StartNodeData,
            NodeType.LLM.value: LLMNodeData,
            NodeType.CODE.value: CodeNodeData,
            NodeType.TEMPLATE_TRANSFORM.value: TemplateTransformNodeData,
            NodeType.IF_ELSE.value: IfElseNodeData,
            NodeType.HTTP_REQUEST.value: HTTPRequestNodeData,
            NodeType.END.value: EndNodeData,
        }
        
        if file_path:
            self.load(file_path)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - RAII cleanup"""
        # Auto-save if file was loaded and modified
        if self.file_path and self.dsl:
            try:
                self.save()
            except Exception:
                pass  # Silent cleanup
    
    # === File Operations ===
    
    def load(self, file_path: str) -> None:
        """Load workflow from YAML file and parse into DSL model"""
        self.file_path = file_path
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        # Parse and validate using DSL model
        self.dsl = DifyWorkflowDSL.model_validate(data)

    def save(self, file_path: Optional[str] = None) -> str:
        """Save workflow to YAML file (serialized from DSL model)"""
        output_path = file_path or self.file_path
        if not output_path:
            raise ValueError("No file path specified")

        if self.dsl is None:
            raise ValueError("No workflow loaded")

        # Serialize pydantic model to dict for YAML output
        workflow_data = self.dsl.model_dump()

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(workflow_data, f,
                      default_flow_style=False,
                      allow_unicode=True,
                      sort_keys=False)

        return output_path
    
    # === Basic Properties ===
    
    @property
    def is_loaded(self) -> bool:
        """Check if workflow is loaded"""
        return self.dsl is not None
    
    @property 
    def nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the workflow (as dicts for backward compatibility)"""
        if self.dsl is None:
            return []
        return [n.model_dump() for n in self.dsl.workflow.graph.nodes]

    @property
    def edges(self) -> List[Dict[str, Any]]:
        """Get all edges in the workflow (as dicts for backward compatibility)"""
        if self.dsl is None:
            return []
        return [e.model_dump() for e in self.dsl.workflow.graph.edges]
    
    # === Workflow Information ===
    
    def get_workflow_info(self) -> WorkflowInfo:
        """Get summary information about the workflow"""
        if self.dsl is None:
            return WorkflowInfo(
                app_name="No workflow loaded",
                description="",
                mode="workflow",
                node_count=0,
                edge_count=0,
                node_types={}
            )
        
        # Count node types
        node_types: Dict[str, int] = {}
        for node in self.nodes:
            node_type = node.get('data', {}).get('type', 'unknown')
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
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Find a node by ID (dict view)"""
        if self.dsl is None:
            return None
        n = self._get_node_model(node_id)
        return n.model_dump() if n else None
    
    def _get_node_model(self, node_id: str) -> Optional[Node]:
        """Get node as pydantic model"""
        if self.dsl is None:
            return None
        for n in self.dsl.workflow.graph.nodes:
            if n.id == node_id:
                return n
        return None
    
    def get_node_connections(self, node_id: str) -> NodeConnection:
        """Get incoming and outgoing connections for a node"""
        incoming: List[str] = []
        outgoing: List[str] = []
        
        if self.dsl is None:
            return NodeConnection(incoming=incoming, outgoing=outgoing)
            
        for edge in self.dsl.workflow.graph.edges:
            if edge.source == node_id:
                outgoing.append(edge.target)
            if edge.target == node_id:
                incoming.append(edge.source)
                
        return NodeConnection(incoming=incoming, outgoing=outgoing)
    
    def get_terminal_nodes(self) -> List[Dict[str, Any]]:
        """Get all terminal nodes (nodes with no outgoing edges)"""
        terminal_nodes: List[Dict[str, Any]] = []
        for node in self.nodes:
            node_id = node.get('id')
            connections = self.get_node_connections(node_id)
            if not connections.outgoing:
                terminal_nodes.append(node)
        return terminal_nodes
    
    # === Node Modification ===
    
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

        if self.dsl is None:
            raise ValueError("No workflow loaded")
            
        # Validate/construct Node via Pydantic and add to model graph
        node_model = Node.model_validate(new_node)
        self.dsl.workflow.graph.nodes.append(node_model)
        # Update edges in model
        self._insert_node_in_edges(after_node_id, node_id,
                                   after_node['data']['type'],
                                   new_node['data']['type'])
        return node_id
    
    def add_node_at_end(self, new_node: Dict[str, Any]) -> str:
        """Add a new node at the end of the workflow"""
        terminal_nodes = self.get_terminal_nodes()
        if not terminal_nodes:
            raise ValueError("No terminal nodes found in workflow")
        terminal_node = terminal_nodes[0]
        return self.add_node_after(terminal_node['id'], new_node)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and reconnect edges"""
        node = self.get_node(node_id)
        if not node:
            return False

        # Get connections
        connections = self.get_node_connections(node_id)

        if self.dsl is None:
            raise ValueError("No workflow loaded")
            
        # Remove node from model
        self.dsl.workflow.graph.nodes = [n for n in self.dsl.workflow.graph.nodes if n.id != node_id]
        # Remove connected edges
        self.dsl.workflow.graph.edges = [
            e for e in self.dsl.workflow.graph.edges
            if e.source != node_id and e.target != node_id
        ]
        # Reconnect edges (bridge the gap)
        for source in connections.incoming:
            for target in connections.outgoing:
                source_node = self.get_node(source)
                target_node = self.get_node(target)
                if source_node and target_node:
                    self._add_edge(source, target,
                                   source_node['data']['type'],
                                   target_node['data']['type'])
        return True
    
    # === Validation ===
    
    def validate_node_data(self, node_type: str, node_data: Dict[str, Any]) -> NodeValidationResult:
        """
        Validate node data against pydantic model
        Returns: NodeValidationResult with validation status and errors
        """
        errors = []
        
        # Get the appropriate model for this node type
        model_class = self._node_models.get(node_type)
        
        if not model_class:
            # Unknown node type, allow it
            return NodeValidationResult(node_id="", is_valid=True, errors=[])
        
        try:
            # Validate using pydantic model
            model_class(**node_data)
            return NodeValidationResult(node_id="", is_valid=True, errors=[])
        except ValidationError as e:
            # Parse validation errors
            for error in e.errors():
                field = ' -> '.join(str(loc) for loc in error['loc'])
                msg = error['msg']
                errors.append(f"{field}: {msg}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return NodeValidationResult(node_id="", is_valid=False, errors=errors)
    
    def validate_workflow(self) -> WorkflowValidationResult:
        """
        Validate entire workflow structure and nodes
        Returns: WorkflowValidationResult with comprehensive validation info
        """
        if self.dsl is None:
            return WorkflowValidationResult(
                is_valid=False,
                structure_errors=["No workflow loaded"],
                node_errors={},
                graph_errors=[]
            )
        
        structure_errors = []
        node_errors = {}
        graph_errors = []
        is_valid = True
        
        # Get workflow data for validation
        workflow_data = self.dsl.model_dump()
        
        # Validate nodes
        nodes = workflow_data.get('workflow', {}).get('graph', {}).get('nodes', [])
        for node in nodes:
            node_id = node.get('id', 'unknown')
            node_data = node.get('data', {})
            node_type = node_data.get('type')
            
            if not node_type:
                node_errors[node_id] = ["Missing node type"]
                is_valid = False
                continue
            
            validation_result = self.validate_node_data(node_type, node_data)
            if not validation_result.is_valid:
                node_errors[node_id] = validation_result.errors
                is_valid = False
        
        # Validate by parsing full DSL using new Pydantic models
        try:
            DifyWorkflowDSL.model_validate(workflow_data)
        except ValidationError as e:
            for error in e.errors():
                field = ' -> '.join(str(loc) for loc in error['loc'])
                msg = error['msg']
                graph_errors.append(f"{field}: {msg}")
            is_valid = False
        except Exception as e:
            graph_errors.append(f"DSL validation error: {str(e)}")
            is_valid = False
        
        return WorkflowValidationResult(
            is_valid=is_valid,
            structure_errors=structure_errors,
            node_errors=node_errors,
            graph_errors=graph_errors
        )
    
    def is_linear_workflow(self) -> LinearityCheck:
        """
        Check if workflow is linear (no branches or merges)
        Returns: LinearityCheck with result and error message
        """
        if self.dsl is None:
            return LinearityCheck(
                is_linear=False,
                error_message="No workflow loaded"
            )
        
        workflow_data = self.dsl.model_dump()
        edges = workflow_data.get('workflow', {}).get('graph', {}).get('edges', [])
        nodes = workflow_data.get('workflow', {}).get('graph', {}).get('nodes', [])
        
        # Count outgoing and incoming edges for each node
        outgoing_count = {}
        incoming_count = {}
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            
            outgoing_count[source] = outgoing_count.get(source, 0) + 1
            incoming_count[target] = incoming_count.get(target, 0) + 1
        
        # Check for branches (multiple outgoing)
        for node_id, count in outgoing_count.items():
            if count > 1:
                return LinearityCheck(
                    is_linear=False,
                    error_message=f"Node '{node_id}' has {count} outgoing edges (branching detected)"
                )
        
        # Check for merges (multiple incoming, except for end node)
        for node_id, count in incoming_count.items():
            if count > 1:
                # Find if this is an end node
                node = next((n for n in nodes if n.get('id') == node_id), None)
                if node and node.get('data', {}).get('type') != 'end':
                    return LinearityCheck(
                        is_linear=False,
                        error_message=f"Node '{node_id}' has {count} incoming edges (merge detected)"
                    )
        
        return LinearityCheck(is_linear=True, error_message=None)
    
    def get_node_schema(self, node_type: str) -> Optional[Dict[str, Any]]:
        """Get JSON schema for a node type"""
        model_class = self._node_models.get(node_type)
        if model_class:
            return model_class.model_json_schema()
        return None
    
    # === Private Helper Methods ===
    
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
        if connections.outgoing:
            next_node = self.get_node(connections.outgoing[0])
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
        if self.dsl is None:
            raise ValueError("No workflow loaded")
            
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