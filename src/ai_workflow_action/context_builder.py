"""
Simple linear workflow context builder
Extracts linear node sequences for AI generation
"""

from typing import Dict, Any, List, Optional


class ContextBuilder:
    """Build simple linear context from workflow for AI generation"""
    
    def extract_linear_sequence(self, workflow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract nodes in linear execution order from start to end
        Returns list of nodes with their data
        """
        nodes = workflow_data.get('workflow', {}).get('graph', {}).get('nodes', [])
        edges = workflow_data.get('workflow', {}).get('graph', {}).get('edges', [])
        
        # Build adjacency map
        adjacency = {}
        for edge in edges:
            adjacency[edge['source']] = edge['target']
        
        # Find start node
        start_node = None
        for node in nodes:
            if node.get('data', {}).get('type') == 'start':
                start_node = node
                break
        
        if not start_node:
            raise ValueError("No start node found in workflow")
        
        # Follow the linear path
        sequence = []
        current_id = start_node.get('id')
        visited = set()
        
        while current_id:
            # Prevent infinite loops
            if current_id in visited:
                raise ValueError(f"Cycle detected at node '{current_id}'")
            visited.add(current_id)
            
            # Find current node
            current_node = next((n for n in nodes if n.get('id') == current_id), None)
            if current_node:
                sequence.append({
                    'id': current_node.get('id'),
                    'data': current_node.get('data', {})
                })
            
            # Move to next node
            current_id = adjacency.get(current_id)
        
        return sequence
    
    def build_context(self, workflow_data: Dict[str, Any], 
                     target_position: Optional[str] = None) -> Dict[str, Any]:
        """
        Build context for AI generation
        
        Args:
            workflow_data: The workflow data
            target_position: Node ID after which to insert (None = end)
        
        Returns:
            Context dictionary with app info and node sequence
        """
        # Extract linear sequence
        full_sequence = self.extract_linear_sequence(workflow_data)
        
        # If target_position specified, truncate sequence
        if target_position:
            truncated_sequence = []
            for node in full_sequence:
                truncated_sequence.append(node)
                if node['id'] == target_position:
                    break
            node_sequence = truncated_sequence
        else:
            node_sequence = full_sequence
        
        # Build context
        context = {
            'app_name': workflow_data.get('app', {}).get('name', 'Unknown'),
            'description': workflow_data.get('app', {}).get('description', ''),
            'mode': workflow_data.get('app', {}).get('mode', 'workflow'),
            'node_sequence': node_sequence
        }
        
        return context
    
    def build_generation_prompt(self, context: Dict[str, Any], 
                               target_node_type: str) -> str:
        """
        Build a simple, effective prompt for node generation
        
        Args:
            context: The workflow context
            target_node_type: Type of node to generate
        
        Returns:
            Prompt string for AI generation
        """
        # Extract just the data fields for clarity
        node_data_sequence = [node['data'] for node in context['node_sequence']]
        
        prompt = f"""You are generating a {target_node_type} node for a Dify workflow.

## Workflow Context
App: {context['app_name']}
Description: {context['description']}

## Previous Nodes in Sequence
The workflow so far consists of these nodes (showing their data fields):

{self._format_node_sequence(node_data_sequence)}

## Your Task
Generate the 'data' field for a new {target_node_type} node that:
1. Logically follows the previous nodes
2. Uses outputs from previous nodes where appropriate (using {{{{#node_id.variable#}}}} syntax)
3. Has a clear purpose that advances the workflow

## Important Notes
- Reference variables from previous nodes using {{{{#node_id.variable#}}}} syntax
- For example: {{{{#start.user_query#}}}} or {{{{#llm-1.text#}}}}
- The data must be valid according to Dify's schema for {target_node_type} nodes
- Include all required fields for the node type

Return ONLY a valid JSON object for the node's data field, no explanation."""
        
        return prompt
    
    def _format_node_sequence(self, node_data_sequence: List[Dict[str, Any]]) -> str:
        """Format node sequence for display in prompt"""
        formatted = []
        for i, data in enumerate(node_data_sequence, 1):
            node_type = data.get('type', 'unknown')
            title = data.get('title', 'Untitled')
            
            # Add key information based on node type
            info = f"{i}. {title} (type: {node_type})"
            
            if node_type == 'start':
                variables = data.get('variables', [])
                if variables:
                    var_names = [v.get('variable') for v in variables]
                    info += f"\n   Inputs: {', '.join(var_names)}"
            
            elif node_type == 'llm':
                model = data.get('model', {})
                info += f"\n   Model: {model.get('provider')}/{model.get('name')}"
                info += f"\n   Temperature: {model.get('completion_params', {}).get('temperature')}"
            
            elif node_type == 'code':
                info += f"\n   Language: {data.get('code_language')}"
                outputs = data.get('outputs', {})
                if outputs:
                    info += f"\n   Outputs: {', '.join(outputs.keys())}"
            
            elif node_type == 'end':
                outputs = data.get('outputs', [])
                if outputs:
                    info += f"\n   Outputs: {len(outputs)} configured"
            
            formatted.append(info)
        
        return '\n'.join(formatted)