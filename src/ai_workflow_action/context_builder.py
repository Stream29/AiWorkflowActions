"""
DifyWorkflowContextBuilder - Specialized context builder for AI operations
Extracts and formats workflow information for AI node generation
Simplified implementation focused on essential functionality
"""

import json
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

from .dsl_file import DifyWorkflowDslFile
from .models import WorkflowContext, NodeInfo


class DifyWorkflowContextBuilder:
    """
    Simplified context builder for AI workflow operations
    Extracts workflow state and builds prompts for AI generation
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
        
        # Extract topologically sorted sequence
        full_sequence = self._extract_topological_sequence(dsl_file)
        
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
        
        # Convert to NodeInfo objects with topology information
        node_info_sequence = [
            NodeInfo(
                id=node['id'],
                title=node['data'].get('title', 'Untitled'),
                type=node['data'].get('type', 'unknown'),
                data=node['data'],
                successor_nodes=node.get('successor_nodes', []),
                predecessor_nodes=node.get('predecessor_nodes', [])
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
        Build a simplified prompt for AI node generation

        Args:
            context: The workflow context
            target_node_type: Type of node to generate
            schema: Optional JSON schema for the node type
            previous_errors: Optional list of previous validation errors for retry

        Returns:
            Prompt string for AI generation
        """
        # Extract node data with outputs for variable reference
        node_sequence = []
        for node in context.node_sequence:
            node_info = {
                'id': node.id,
                'type': node.type,
                'title': node.title,
                'successors': node.successor_nodes,
                'predecessors': node.predecessor_nodes
            }

            # Add common outputs for variable reference
            outputs = []
            if node.type == 'start':
                outputs = [v.get('variable') for v in node.data.get('variables', [])]
            elif node.type == 'code':
                outputs = list(node.data.get('outputs', {}).keys())
            elif node.type == 'llm':
                outputs = ['text']
            elif node.type == 'http-request':
                outputs = ['body', 'status_code']
            elif node.type == 'parameter-extractor':
                outputs = [p.get('name') for p in node.data.get('parameters', []) if p.get('name')]

            if outputs:
                node_info['outputs'] = outputs
            node_sequence.append(node_info)

        # Build prompt
        prompt = f"""Generate configuration for a {target_node_type} node in a Dify workflow.

## Workflow Context
App: {context.app_name}
Description: {context.description}

## Workflow Nodes (Topologically Sorted)
{json.dumps(node_sequence, indent=2)}

Note: Each node includes 'successors' and 'predecessors' arrays showing the workflow structure.
The nodes are arranged in topological order supporting complex workflows with branches and merges.

## Variable Reference Syntax
Reference outputs using: {{{{#node_id.variable#}}}}"""

        # Add available variables
        for node in node_sequence:
            if node.get('outputs'):
                for output in node['outputs']:
                    prompt += f"\n- {{{{#{node['id']}.{output}#}}}}"

        # Add schema if provided
        if schema:
            prompt += f"\n\n## Schema\n{json.dumps(schema, indent=2)}"

        # Add error feedback
        if previous_errors:
            prompt += "\n\n## Previous Errors\n"
            for error in previous_errors:
                prompt += f"- {error}\n"
            prompt += "Fix these issues in your response."

        # Final instructions
        prompt += f"""

## Instructions
1. Generate a valid 'data' field for a {target_node_type} node
2. Use variables from previous nodes where logical
3. Follow the schema requirements exactly
4. Create meaningful titles and descriptions
5. Return ONLY the JSON object, no markdown or explanation"""

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
            return {"recommendations": ["start"]}

        last_node = context.node_sequence[-1]
        last_type = last_node.type

        # Simple recommendations based on last node type
        recommendations = {
            'start': ['llm', 'code', 'http-request'],
            'llm': ['code', 'end', 'if-else'],
            'code': ['end', 'llm'],
            'http-request': ['code', 'llm', 'end'],
        }.get(last_type, ['end'])

        return {
            "last_node": {"id": last_node.id, "type": last_type, "title": last_node.title},
            "is_complete": last_type == 'end',
            "recommendations": recommendations
        }
    
    # === Private Helper Methods ===
    
    def _extract_topological_sequence(self, dsl_file: DifyWorkflowDslFile) -> List[Dict[str, Any]]:
        """
        Extract nodes using topological sorting to handle complex workflows
        Returns list of nodes with their data and topology information
        """
        nodes = dsl_file.nodes
        edges = dsl_file.edges

        # Build node lookup
        node_dict = {node.get('id'): node for node in nodes}

        # Build adjacency lists
        adjacency: Dict[str, List[str]] = defaultdict(list)  # node_id -> [successor_ids]
        reverse_adjacency: Dict[str, List[str]] = defaultdict(list)  # node_id -> [predecessor_ids]
        in_degree: Dict[str, int] = defaultdict(int)

        # Initialize all nodes with zero in-degree
        for node in nodes:
            node_id = node.get('id')
            if node_id:
                in_degree[node_id] = 0

        # Build graph and calculate in-degrees
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                adjacency[source].append(target)
                reverse_adjacency[target].append(source)
                in_degree[target] += 1

        # Kahn's algorithm for topological sorting
        queue = deque()

        # Start with nodes that have no incoming edges
        for node_id, degree in in_degree.items():
            if degree == 0:
                queue.append(node_id)

        result = []

        while queue:
            current_id = queue.popleft()
            current_node = node_dict.get(current_id)

            if current_node:
                # Add topology information to node data
                node_data = {
                    'id': current_node.get('id'),
                    'data': current_node.get('data', {}),
                    'successor_nodes': adjacency.get(current_id, []),
                    'predecessor_nodes': reverse_adjacency.get(current_id, [])
                }
                result.append(node_data)

            # Reduce in-degree for all successors
            for neighbor in adjacency.get(current_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(nodes):
            # Find nodes not in result (part of cycle)
            processed_ids = {node['id'] for node in result}
            unprocessed = [node.get('id') for node in nodes if node.get('id') not in processed_ids]
            raise ValueError(f"Workflow contains cycles involving nodes: {unprocessed}")

        return result