"""
DifyWorkflowContextBuilder - Specialized context builder for AI operations
Extracts and formats workflow information for AI node generation
Focused on creating high-quality prompts and context
"""

from typing import Dict, Any, List, Optional

from dsl_model import DifyWorkflowDSL
from .dsl_file import DifyWorkflowDslFile
from .models import WorkflowContext, NodeInfo


class DifyWorkflowContextBuilder:
    """
    Specialized context builder for AI workflow operations
    Responsible for extracting workflow state and building AI-friendly context
    """
    
    def __init__(self):
        """Initialize the context builder"""
        pass
    
    def build_context(self, 
                     dsl_file: DifyWorkflowDslFile, 
                     target_position: Optional[str] = None) -> WorkflowContext:
        """
        Build context for AI generation from workflow file
        
        Args:
            dsl_file: The workflow file to extract context from
            target_position: Node ID after which to insert (None = end)
        
        Returns:
            WorkflowContext with app info and node sequence
        """
        if not dsl_file.is_loaded:
            raise ValueError("Workflow file is not loaded")
        
        # Extract linear sequence
        full_sequence = self._extract_linear_sequence(dsl_file)
        
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
        
        # Convert to NodeInfo objects
        node_info_sequence = [
            NodeInfo(
                id=node['id'],
                title=node['data'].get('title', 'Untitled'),
                type=node['data'].get('type', 'unknown'),
                data=node['data']
            )
            for node in node_sequence
        ]
        
        # Get workflow info for context
        workflow_info = dsl_file.get_workflow_info()
        
        # Build context
        return WorkflowContext(
            app_name=workflow_info.app_name,
            description=workflow_info.description,
            mode=workflow_info.mode,
            node_sequence=node_info_sequence
        )
    
    def build_generation_prompt(self, 
                               context: WorkflowContext, 
                               target_node_type: str,
                               schema: Optional[Dict[str, Any]] = None,
                               previous_errors: Optional[List[str]] = None) -> str:
        """
        Build a comprehensive prompt for AI node generation
        
        Args:
            context: The workflow context
            target_node_type: Type of node to generate
            schema: Optional JSON schema for the node type
            previous_errors: Optional list of previous validation errors for retry
        
        Returns:
            Detailed prompt string for AI generation
        """
        # Extract node data sequence for analysis
        node_sequence = []
        for node in context.node_sequence:
            node_info = {
                'id': node.id,
                'type': node.type,
                'title': node.title
            }
            
            # Add key outputs/variables for reference
            if node.type == 'start':
                variables = node.data.get('variables', [])
                node_info['outputs'] = [v.get('variable') for v in variables]
            elif node.type == 'code':
                node_info['outputs'] = list(node.data.get('outputs', {}).keys())
            elif node.type == 'llm':
                node_info['outputs'] = ['text']  # LLM nodes output 'text'
            elif node.type == 'http-request':
                node_info['outputs'] = ['body', 'status_code']  # Common HTTP outputs
            elif node.type == 'parameter-extractor':
                parameters = node.data.get('parameters', [])
                node_info['outputs'] = [p.get('name') for p in parameters if p.get('name')]
            
            node_sequence.append(node_info)
        
        # Build the comprehensive prompt
        prompt = f"""Generate configuration for a {target_node_type} node in a Dify workflow.

## Workflow Context
App: {context.app_name}
Description: {context.description}

## Node Sequence
Previous nodes in the workflow:
{self._format_node_sequence_json(node_sequence)}

## Available Variables
You can reference outputs from previous nodes using {{{{#node_id.variable#}}}} syntax.
Examples based on the nodes above:"""
        
        # Add variable references
        for node in node_sequence:
            if node.get('outputs'):
                for output in node['outputs']:
                    prompt += f"\n- {{{{#{node['id']}.{output}#}}}}"
        
        # Add schema if provided
        if schema:
            prompt += f"\n\n## Node Schema\nThe {target_node_type} node should follow this structure:\n{self._format_schema_for_ai(schema)}"
        
        # Add error feedback if retrying
        if previous_errors:
            prompt += "\n\n## Previous Validation Errors\nThe previous attempt had these errors:\n"
            for error in previous_errors:
                prompt += f"- {error}\n"
            prompt += "\nPlease fix these issues in your response."
        
        # Add specific instructions based on node type
        prompt += self._get_node_type_instructions(target_node_type)
        
        # Add final instructions
        prompt += f"""

## Instructions
1. Generate a valid 'data' field for a {target_node_type} node
2. Use appropriate variables from previous nodes where it makes logical sense
3. Ensure all required fields are included according to the schema
4. Make the node's purpose clear and logical within the workflow
5. Create meaningful titles and descriptions

Return ONLY the JSON object for the data field, no explanation or markdown."""
        
        return prompt
    
    def analyze_workflow_completion(self, context: WorkflowContext) -> Dict[str, Any]:
        """
        Analyze workflow to suggest next logical steps
        
        Args:
            context: Current workflow context
        
        Returns:
            Analysis with recommendations for next nodes
        """
        if not context.node_sequence:
            return {"status": "empty", "recommendations": ["start"]}
        
        last_node = context.node_sequence[-1]
        last_type = last_node.type
        
        # Basic completion analysis
        analysis = {
            "last_node": {
                "id": last_node.id,
                "type": last_type,
                "title": last_node.title
            },
            "is_complete": last_type == 'end',
            "recommendations": self._get_node_recommendations(last_type),
            "workflow_stage": self._analyze_workflow_stage(context)
        }
        
        return analysis
    
    # === Private Helper Methods ===
    
    def _extract_linear_sequence(self, dsl_file: DifyWorkflowDslFile) -> List[Dict[str, Any]]:
        """
        Extract nodes in linear execution order from start to end
        Returns list of nodes with their data
        """
        nodes = dsl_file.nodes
        edges = dsl_file.edges
        
        # Build adjacency map (assume at most one outgoing in linear flows)
        adjacency: Dict[str, str] = {}
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
        sequence: List[Dict[str, Any]] = []
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
    
    def _format_node_sequence_json(self, node_sequence: List[Dict[str, Any]]) -> str:
        """Format node sequence as JSON for AI consumption"""
        import json
        return json.dumps(node_sequence, indent=2)
    
    def _format_schema_for_ai(self, schema: Dict[str, Any]) -> str:
        """Simplify and format JSON schema for better AI understanding"""
        import json
        simplified = {}
        
        # Extract required fields
        if 'required' in schema:
            simplified['required_fields'] = schema['required']
        
        # Extract property types
        if 'properties' in schema:
            simplified['fields'] = {}
            for field, field_schema in schema['properties'].items():
                field_type = field_schema.get('type', 'any')
                description = field_schema.get('description', '')
                
                simplified['fields'][field] = {
                    'type': field_type,
                    'description': description[:100] if description else 'No description'
                }
                
                # Add enum values if present
                if 'enum' in field_schema:
                    simplified['fields'][field]['allowed_values'] = field_schema['enum']
        
        return json.dumps(simplified, indent=2)
    
    def _get_node_type_instructions(self, node_type: str) -> str:
        """Get specific instructions based on node type"""
        instructions = {
            'llm': """
## LLM Node Specific Guidelines
- Use a clear, specific prompt that builds on previous context
- Reference variables from previous nodes appropriately
- Set reasonable temperature (0.1-0.7) based on creativity needs
- Choose appropriate model (gpt-3.5-turbo for simple tasks, gpt-4 for complex)""",
            
            'code': """
## Code Node Specific Guidelines  
- Write clean, focused code that performs a specific transformation
- Use variables from previous nodes as inputs
- Define clear outputs that other nodes can use
- Include proper error handling
- Use appropriate programming language (Python recommended)""",
            
            'http-request': """
## HTTP Request Node Guidelines
- Build URLs and parameters using previous node outputs
- Set appropriate headers and authentication
- Handle common HTTP methods (GET, POST, PUT, DELETE)
- Define expected response structure""",
            
            'end': """
## End Node Guidelines
- Define clear outputs that summarize the workflow results
- Use descriptive names for output variables  
- Include relevant data from the workflow execution""",
            
            'if-else': """
## Conditional Node Guidelines
- Create logical conditions based on previous node outputs
- Use clear, readable condition expressions
- Ensure both branches handle the logic appropriately""",
        }
        
        return instructions.get(node_type, "")
    
    def _get_node_recommendations(self, last_node_type: str) -> List[str]:
        """Get recommended next node types based on the last node"""
        recommendations = {
            'start': ['llm', 'code', 'http-request', 'variable-assigner'],
            'llm': ['code', 'end', 'if-else', 'variable-assigner', 'parameter-extractor'],
            'code': ['end', 'llm', 'if-else', 'variable-assigner'],
            'http-request': ['code', 'llm', 'parameter-extractor', 'end'],
            'variable-assigner': ['llm', 'code', 'end', 'if-else'],
            'parameter-extractor': ['llm', 'code', 'end'],
            'if-else': ['llm', 'code', 'end'],
        }
        return recommendations.get(last_node_type, ['end'])
    
    def _analyze_workflow_stage(self, context: WorkflowContext) -> str:
        """Analyze what stage the workflow is in"""
        node_types = [node.type for node in context.node_sequence]
        
        if len(node_types) == 1 and node_types[0] == 'start':
            return "initialization"
        elif 'end' in node_types:
            return "complete"
        elif any(t in ['llm', 'code'] for t in node_types):
            return "processing" 
        else:
            return "building"