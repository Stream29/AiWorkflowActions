"""
DifyWorkflowContextBuilder - Specialized context builder for AI operations
Extracts and formats workflow information for AI node generation
Simplified implementation focused on essential functionality
"""

import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Type

from pydantic import BaseModel

from .dsl_file import DifyWorkflowDslFile
from .models import WorkflowContext, NodeInfo, TopologyNodeData, NodeOutputInfo


class DifyWorkflowContextBuilder:
    """
    Simplified context builder for AI workflow operations
    Extracts workflow state and builds prompts for AI generation
    """

    def __init__(self):
        """Initialize the context builder"""
        pass

    @staticmethod
    def build_context(
            dsl_file: DifyWorkflowDslFile,
            target_position: str
    ) -> WorkflowContext:
        """
        Build context for AI generation from workflow file
        
        Args:
            dsl_file: The workflow file to extract context from
            target_position: Node ID after which to insert (None = end)
        
        Returns:
            WorkflowContext with app info and node sequence
        """

        # Extract topologically sorted sequence
        full_sequence = DifyWorkflowContextBuilder.extract_topological_sequence(dsl_file)

        # If target_position specified, truncate sequence
        if target_position:
            truncated_sequence = []
            for node in full_sequence:
                truncated_sequence.append(node)
                if node.id == target_position:
                    break
            node_sequence = truncated_sequence
        else:
            node_sequence = full_sequence

        # Convert to NodeInfo objects with topology information
        node_info_sequence = [
            NodeInfo(
                id=node.id,
                title=node.data.title,
                type=node.data.type,
                data=node.data,
                successor_nodes=node.successor_nodes,
                predecessor_nodes=node.predecessor_nodes
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

    @staticmethod
    def build_generation_prompt(
            context: WorkflowContext,
            target_node_type: str,
            node_model_class: Optional[Type[BaseModel]] = None,
    ) -> str:
        # Extract node data with outputs for variable reference using typed models
        node_output_sequence: List[NodeOutputInfo] = []
        for node in context.node_sequence:
            node_output_info = NodeOutputInfo(
                id=node.id,
                type=node.type,
                title=node.title,
                successors=node.successor_nodes,
                predecessors=node.predecessor_nodes,
                outputs=[]
            )
            node_output_sequence.append(node_output_info)

        # Build prompt
        prompt = f"""Generate configuration for a {target_node_type} node in a Dify workflow.

## Workflow Context
App: {context.app_name}
Description: {context.description}

## Workflow Nodes (Topologically Sorted)
{json.dumps([node.model_dump() for node in node_output_sequence], indent=2)}

Note: Each node includes 'successors' and 'predecessors' arrays showing the workflow structure.
The nodes are arranged in topological order supporting complex workflows with branches and merges.

## Variable Reference Syntax
Reference outputs using: {{{{#node_id.variable#}}}}"""

        # Add available variables
        for node in node_output_sequence:
            if node.outputs:
                for output in node.outputs:
                    prompt += f"\n- {{{{#{node.id}.{output}#}}}}"

        # Add schema if provided
        if node_model_class:
            schema = node_model_class.model_json_schema()
            prompt += f"\n\n## Schema\n{json.dumps(schema, indent=2)}"

        # Final instructions
        prompt += f"""

## Instructions
1. Generate a valid 'data' field for a {target_node_type} node
2. Use variables from previous nodes where logical
3. Follow the schema requirements exactly
4. Create meaningful titles and descriptions
5. Return ONLY the JSON object, no markdown or explanation"""

        return prompt

    @staticmethod
    def extract_topological_sequence(dsl_file: DifyWorkflowDslFile) -> List[TopologyNodeData]:
        """
        Extract nodes using topological sorting to handle complex workflows
        Returns list of nodes with their data and topology information
        """
        nodes = dsl_file.dsl.workflow.graph.nodes
        edges = dsl_file.dsl.workflow.graph.edges

        # Build node lookup
        node_dict = {node.id: node for node in nodes}

        # Build adjacency lists
        adjacency: Dict[str, List[str]] = defaultdict(list)  # node_id -> [successor_ids]
        reverse_adjacency: Dict[str, List[str]] = defaultdict(list)  # node_id -> [predecessor_ids]
        in_degree: Dict[str, int] = defaultdict(int)

        # Initialize all nodes with zero in-degree
        for node in nodes:
            in_degree[node.id] = 0

        # Build graph and calculate in-degrees
        for edge in edges:
            source = edge.source
            target = edge.target
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
                # Create typed topology node data
                topology_node = TopologyNodeData(
                    id=current_node.id,
                    data=current_node.data,
                    successor_nodes=adjacency.get(current_id, []),
                    predecessor_nodes=reverse_adjacency.get(current_id, [])
                )
                result.append(topology_node)

            # Reduce in-degree for all successors
            for neighbor in adjacency.get(current_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.appendleft(neighbor)

        # Check for cycles
        if len(result) != len(nodes):
            # Find nodes not in result (part of cycle)
            processed_ids = {node.id for node in result}
            unprocessed = [node.id for node in nodes if node.id not in processed_ids]
            raise ValueError(f"Workflow contains cycles involving nodes: {unprocessed}")

        return result
