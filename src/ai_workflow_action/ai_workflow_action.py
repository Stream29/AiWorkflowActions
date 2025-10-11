from typing import Type, Optional

from anthropic import Anthropic
from dsl_model import NodeType, BaseNodeData
from .context_builder import DifyWorkflowContextBuilder
from .dsl_file import DifyWorkflowDslFile
from .models import WorkflowContext, NodeInfo
from .node_type_util import NodeTypeUtil
from .config_loader import ConfigLoader


class AiWorkflowAction:
    def __init__(self, dsl_file: DifyWorkflowDslFile):
        """
        Initialize AI workflow action

        Args:
            dsl_file: Workflow file to work with
        """
        # Get configuration from global singleton
        config = ConfigLoader.get_config()

        # Initialize AI resources (RAII pattern for API client)
        self.client = Anthropic(api_key=config.api.anthropic_api_key)
        self.model = config.models.generation

        # Initialize workflow components
        self.dsl_file = dsl_file

    def generate_node(
            self,
            after_node_id: str,
            node_type: NodeType,
            user_message: Optional[str] = None,
    ) -> str:
        """
        Generate and add a new node using AI

        Args:
            after_node_id: Node ID after which to insert the new node
            node_type: Type of node to generate
            user_message: Optional user intent/instruction for generation

        Returns:
            New node ID if successful, None otherwise
        """
        # Resolve after_node information
        after_node = self._get_node_info(after_node_id)
        if not after_node:
            raise ValueError(f"Node '{after_node_id}' not found in workflow")

        context = DifyWorkflowContextBuilder.build_context(
            self.dsl_file,
            target_position=after_node_id
        )
        node_data_model = NodeTypeUtil.get_node_data_model(NodeType(node_type))
        node_data = self._generate_node_data(
            node_type=node_type,
            context=context,
            node_data_model=node_data_model,
            after_node=after_node,
            user_message=user_message,
        )
        return self.dsl_file.add_node_after(after_node_id, node_data)

    def _get_node_info(self, node_id: str) -> Optional[NodeInfo]:
        """Get NodeInfo for a given node_id"""

        # Get the raw node from DSL
        raw_node = self.dsl_file.get_node(node_id)
        if not raw_node:
            return None

        # Get node connections for predecessor/successor info
        connections = self.dsl_file.get_node_connections(node_id)

        # Build NodeInfo
        return NodeInfo(
            id=raw_node.id,
            title=raw_node.data.title,
            type=raw_node.data.type,
            data=raw_node.data,
            successor_nodes=connections.outgoing,
            predecessor_nodes=connections.incoming,
        )

    def _generate_node_data(
            self,
            node_type: NodeType,
            context: WorkflowContext,
            node_data_model: Type[BaseNodeData],
            after_node: NodeInfo,
            user_message: Optional[str] = None,
    ) -> BaseNodeData:
        prompt = DifyWorkflowContextBuilder.build_generation_prompt(
            context=context,
            target_node_type=node_type,
            after_node=after_node,
            node_model_class=node_data_model,
            user_message=user_message,
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_response = response.content[0].text
        cleaned_json = _clean_json(raw_response)
        return node_data_model.model_validate_json(cleaned_json)


def _clean_json(content: str) -> str:
    if not content or not content.strip():
        raise ValueError("AI returned empty response")
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    content = content.strip()
    start = content.find('{')
    end = content.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No valid JSON object found in response: {content[:200]}...")
    json_content = content[start:end + 1]
    return json_content
