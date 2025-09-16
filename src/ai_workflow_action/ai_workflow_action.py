"""
AiWorkflowAction - Third layer abstraction
Provides AI-powered workflow operations using DifyWorkflowDslFile
Manages Anthropic API resources and advanced workflow manipulation
"""

import json
from typing import Optional, List, Type

from anthropic import Anthropic
from pydantic import BaseModel

from .context_builder import DifyWorkflowContextBuilder
from .dsl_file import DifyWorkflowDslFile
from .models import WorkflowContext


class AiWorkflowAction:
    """
    Third layer: AI-powered workflow operations
    Manages Anthropic API resources and provides intelligent workflow manipulation
    Uses DifyWorkflowDslFile for basic operations and DifyWorkflowContextBuilder for AI context
    """

    def __init__(self,
                 api_key: str,
                 dsl_file: Optional[DifyWorkflowDslFile] = None,
                 model: str = "claude-sonnet-4-20250520"):
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

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup API resources"""
        # Cleanup any API resources if needed
        pass

    # === Workflow File Management ===

    def load_workflow(self, file_path: str) -> None:
        """Load a workflow file"""
        if self.dsl_file is None:
            self.dsl_file = DifyWorkflowDslFile()
        self.dsl_file.load(file_path)

    def save_workflow(self, file_path: Optional[str] = None) -> str:
        """Save the current workflow"""
        if self.dsl_file is None:
            raise ValueError("No workflow loaded")
        return self.dsl_file.save(file_path)

    @property
    def is_workflow_loaded(self) -> bool:
        """Check if a workflow is loaded"""
        return self.dsl_file is not None and self.dsl_file.is_loaded

    # === AI-Powered Node Generation ===

    def generate_node(
            self,
            after_node_id: str,
            node_type: str,
            max_attempts: int = 3
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
        if not self.is_workflow_loaded:
            raise ValueError("No workflow loaded")
        context = self.context_builder.build_context(
            self.dsl_file,
            target_position=after_node_id
        )

        # Get model class for the node type
        node_model_class = self.dsl_file.get_node_model_class(node_type)

        # Generate with retry on validation failure
        previous_errors = None
        for attempt in range(max_attempts):
            try:
                # Generate node data using AI
                node_data_obj = self._generate_node_data(
                    node_type, context, node_model_class, previous_errors
                )

                # Create complete NodeData and add to workflow
                from dsl_model.nodes import NodeData
                complete_node_data = {'type': node_type, **node_data_obj.model_dump()} if hasattr(node_data_obj,
                                                                                                  'model_dump') else {
                    'type': node_type, **node_data_obj}
                final_node_data = NodeData.model_validate(complete_node_data)

                # Add to workflow
                node_id = self.dsl_file.add_node_after(after_node_id, final_node_data)
                return node_id

            except Exception as e:
                previous_errors = [str(e)]

        # All attempts failed
        return None

    def auto_generate_next_node(self, node_type: str) -> Optional[str]:
        """
        Automatically generate and add the next most suitable node

        Args:
            node_type: Specific node type to generate

        Returns:
            New node ID if successful, None otherwise
        """
        if not self.is_workflow_loaded:
            raise ValueError("No workflow loaded")

        # Check if workflow is linear
        linearity_check = self.dsl_file.is_linear_workflow()
        if not linearity_check.is_linear:
            raise ValueError(f"AI generation requires linear workflow: {linearity_check.error_message}")

        # Find terminal nodes
        terminal_nodes = self.dsl_file.get_terminal_nodes()
        if not terminal_nodes:
            raise ValueError("No terminal nodes found in workflow")

        if len(terminal_nodes) > 1:
            # Use the first terminal node
            pass

        terminal_node = terminal_nodes[0]
        terminal_id = terminal_node.id
        terminal_type = terminal_node.data.type

        # Generate and add the node
        return self.generate_node(terminal_id, node_type)

    def generate_workflow_extension(self, target_nodes: List[str]) -> List[str]:
        """
        Generate multiple nodes to extend the workflow
        
        Args:
            target_nodes: List of node types to generate in sequence
        
        Returns:
            List of generated node IDs
        """
        if not self.is_workflow_loaded:
            raise ValueError("No workflow loaded")

        generated_ids = []

        # Find current terminal node
        terminal_nodes = self.dsl_file.get_terminal_nodes()
        if not terminal_nodes:
            raise ValueError("No terminal nodes found in workflow")

        current_terminal = terminal_nodes[0].id

        # Generate nodes in sequence
        for node_type in target_nodes:
            node_id = self.generate_node(current_terminal, node_type)
            if node_id:
                generated_ids.append(node_id)
                current_terminal = node_id
            else:
                break

        return generated_ids

    # === Private AI Methods ===

    def _generate_node_data(self,
                            node_type: str,
                            context: WorkflowContext,
                            node_model_class: Optional[Type[BaseModel]],
                            previous_errors: Optional[List[str]]):
        """
        Generate node data using Claude API

        Args:
            node_type: Type of node to generate
            context: Workflow context
            node_model_class: Pydantic model class for validation
            previous_errors: Optional previous validation errors for retry

        Returns:
            Validated node data object
        """
        # Build comprehensive prompt
        prompt = self.context_builder.build_generation_prompt(
            context, node_type, node_model_class, previous_errors
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract JSON from response
            content = response.content[0].text

            if not content or not content.strip():
                raise ValueError("AI returned empty response")

            # Clean up any markdown formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # Further cleanup - remove any leading/trailing text
            content = content.strip()

            # Find JSON object bounds
            start = content.find('{')
            end = content.rfind('}')

            if start == -1 or end == -1:
                raise ValueError(f"No valid JSON object found in response: {content[:200]}...")

            json_content = content[start:end + 1]

            # Parse JSON and validate with model
            raw_data = json.loads(json_content)
            if node_model_class:
                return node_model_class.model_validate(raw_data)
            else:
                # Fallback for unknown node types
                return raw_data

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse generated JSON: {e}\\nContent: {content[:200] if 'content' in locals() else 'No content'}")
        except Exception as e:
            raise RuntimeError(f"AI generation failed: {e}")
