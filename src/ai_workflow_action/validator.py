"""
Centralized validation using simplified Dify's pydantic models
Single source of truth for all validation logic
"""

from typing import Dict, Any, List, Tuple, Optional
from pydantic import ValidationError

# Import simplified Dify models
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
from dsl_model.dsl import DifyWorkflowDSL


class Validator:
    """Centralized validation for workflows and nodes"""
    
    def __init__(self):
        # Map node types to their pydantic model classes
        self.node_models = {
            NodeType.START.value: StartNodeData,
            NodeType.LLM.value: LLMNodeData,
            NodeType.CODE.value: CodeNodeData,
            NodeType.TEMPLATE_TRANSFORM.value: TemplateTransformNodeData,
            NodeType.IF_ELSE.value: IfElseNodeData,
            NodeType.HTTP_REQUEST.value: HTTPRequestNodeData,
            NodeType.END.value: EndNodeData,
        }
    
    def validate_node_data(self, node_type: str, node_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate node data against pydantic model
        Returns: (is_valid, error_messages)
        """
        errors = []
        
        # Get the appropriate model for this node type
        model_class = self.node_models.get(node_type)
        
        if not model_class:
            # Unknown node type, allow it
            return True, []
        
        try:
            # Validate using pydantic model
            model_class(**node_data)
            return True, []
        except ValidationError as e:
            # Parse validation errors
            for error in e.errors():
                field = ' -> '.join(str(loc) for loc in error['loc'])
                msg = error['msg']
                errors.append(f"{field}: {msg}")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return False, errors
    
    def validate_workflow(self, workflow_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate entire workflow structure and nodes
        Returns: (is_valid, validation_results)
        """
        results = {
            'is_valid': True,
            'structure_errors': [],
            'node_errors': {},
            'graph_errors': []
        }
        
        # Validate structure
        if 'app' not in workflow_data:
            results['structure_errors'].append("Missing 'app' section")
            results['is_valid'] = False
        
        if 'workflow' not in workflow_data:
            results['structure_errors'].append("Missing 'workflow' section")
            results['is_valid'] = False
            return results['is_valid'], results
        
        if 'graph' not in workflow_data.get('workflow', {}):
            results['structure_errors'].append("Missing 'workflow.graph' section")
            results['is_valid'] = False
            return results['is_valid'], results
        
        # Validate nodes
        nodes = workflow_data.get('workflow', {}).get('graph', {}).get('nodes', [])
        for node in nodes:
            node_id = node.get('id', 'unknown')
            node_data = node.get('data', {})
            node_type = node_data.get('type')
            
            if not node_type:
                results['node_errors'][node_id] = ["Missing node type"]
                results['is_valid'] = False
                continue
            
            is_valid, errors = self.validate_node_data(node_type, node_data)
            if not is_valid:
                results['node_errors'][node_id] = errors
                results['is_valid'] = False
        
        # Validate by parsing full DSL using new Pydantic models
        try:
            DifyWorkflowDSL(**workflow_data)
        except ValidationError as e:
            for error in e.errors():
                field = ' -> '.join(str(loc) for loc in error['loc'])
                msg = error['msg']
                results['graph_errors'].append(f"{field}: {msg}")
            results['is_valid'] = False
        except Exception as e:
            results['graph_errors'].append(f"DSL validation error: {str(e)}")
            results['is_valid'] = False
        
        return results['is_valid'], results
    
    def is_linear_workflow(self, workflow_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if workflow is linear (no branches or merges)
        Returns: (is_linear, error_message)
        """
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
                return False, f"Node '{node_id}' has {count} outgoing edges (branching detected)"
        
        # Check for merges (multiple incoming, except for end node)
        for node_id, count in incoming_count.items():
            if count > 1:
                # Find if this is an end node
                node = next((n for n in nodes if n.get('id') == node_id), None)
                if node and node.get('data', {}).get('type') != 'end':
                    return False, f"Node '{node_id}' has {count} incoming edges (merge detected)"
        
        return True, None
    
    def get_node_schema(self, node_type: str) -> Optional[Dict[str, Any]]:
        """Get JSON schema for a node type"""
        model_class = self.node_models.get(node_type)
        if model_class:
            return model_class.model_json_schema()
        return None