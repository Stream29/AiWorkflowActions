from typing import Optional, List, Type

from anthropic import Anthropic

from dsl_model import NodeType, BaseNodeData
from .context_builder import DifyWorkflowContextBuilder
from .dsl_file import DifyWorkflowDslFile
from .models import WorkflowContext
from .node_type_util import NodeTypeUtil


class AiWorkflowAction:
    def __init__(
            self,
            api_key: str,
            dsl_file: DifyWorkflowDslFile,
            model: str = "claude-sonnet-4-20250520"
    ):
        """
        Initialize AI workflow action
        
        Args:
            api_key: Anthropic API key
            dsl_file: Optional workflow file to work with
            model: Claude model to use for generation
        """
        # Initialize AI resources (RAII pattern for API client)
        self.client = Anthropic(api_key=api_key)
        self.model = model

        # Initialize workflow components
        self.dsl_file = dsl_file
        self.context_builder = DifyWorkflowContextBuilder()

    def generate_node(
            self,
            after_node_id: str,
            node_type: NodeType,
    ) -> Optional[str]:
        """
        Generate and add a new node using AI
        
        Args:
            after_node_id: Node ID after which to insert the new node
            node_type: Type of node to generate
            max_attempts: Maximum number of generation attempts
        
        Returns:
            New node ID if successful, None otherwise
        """
        context = self.context_builder.build_context(
            self.dsl_file,
            target_position=after_node_id
        )
        node_data_model = NodeTypeUtil.get_node_data_model(NodeType(node_type))
        node_data = self._generate_node_data(
            node_type=node_type,
            context=context,
            node_data_model=node_data_model,
        )
        return self.dsl_file.add_node_after(after_node_id, node_data)

    def _generate_node_data(
            self,
            node_type: NodeType,
            context: WorkflowContext,
            node_data_model: Type[BaseNodeData],
    ) -> BaseNodeData:
        prompt = self.context_builder.build_generation_prompt(
            context=context,
            target_node_type=node_type,
            node_model_class=node_data_model,
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
