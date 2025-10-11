"""
Phase 3: Node Generator
Generates nodes using AiWorkflowAction API (reuses existing code).
"""

import time
import random
import traceback
from typing import List
from ai_workflow_action import DifyWorkflowDslFile, AiWorkflowAction
from dsl_model import NodeType, NodeData
from ..models import Phase2Dataset, Phase2Sample, Phase3Dataset, Phase3Sample
from ai_workflow_action.config_loader import ConfigLoader


class NodeGenerator:
    """Node generator with retry mechanism (reuses AiWorkflowAction API)"""

    def __init__(self):
        config = ConfigLoader.get_config()
        self.retry_config = config.evaluation.retry

    def generate(self, phase2_data: Phase2Dataset) -> Phase3Dataset:
        """
        Generate nodes using AiWorkflowAction.generate_node() API.

        Args:
            phase2_data: Phase 2 dataset

        Returns:
            Phase 3 dataset with actual_output populated
        """
        phase3_samples: List[Phase3Sample] = []

        for i, p2_sample in enumerate(phase2_data.samples):
            print(f"  [{i+1}/{len(phase2_data.samples)}] Sample {p2_sample.sample_id}...", end=" ")

            # Generate with retry
            actual_output, generated_id, error = self._generate_with_retry(p2_sample)

            phase3_sample = Phase3Sample(
                sample_id=p2_sample.sample_id,
                source_file=p2_sample.source_file,
                masked_workflow=p2_sample.masked_workflow,
                node_type=p2_sample.node_type,
                after_node_id=p2_sample.after_node_id,
                app_name=p2_sample.app_name,
                app_description=p2_sample.app_description,
                user_message=p2_sample.user_message,
                actual_output=actual_output,
                generated_node_id=generated_id,
                generation_error=error
            )
            phase3_samples.append(phase3_sample)

            if error:
                print(f"✗ ({error[:30]})")
            else:
                print("✓")

        return Phase3Dataset(samples=phase3_samples, metadata=phase2_data.metadata)

    def _generate_with_retry(self, p2_sample: Phase2Sample) -> tuple[NodeData, str, str]:
        """
        Generate node with retry mechanism.

        Returns:
            Tuple of (actual_output, generated_node_id, error_message)
        """
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Create temporary workflow from memory
                temp_wf = DifyWorkflowDslFile.__new__(DifyWorkflowDslFile)
                temp_wf.dsl = p2_sample.masked_workflow.dsl
                temp_wf.file_path = None

                # Call existing API (automatically handles prompt building, etc.)
                ai_action = AiWorkflowAction(dsl_file=temp_wf)

                new_node_id = ai_action.generate_node(
                    after_node_id=p2_sample.after_node_id,
                    node_type=NodeType(p2_sample.node_type),
                    user_message=p2_sample.user_message
                )

                # Extract generated node
                generated_node = temp_wf.get_node(new_node_id)
                if not generated_node:
                    raise ValueError(f"Generated node {new_node_id} not found")

                return (generated_node.data, new_node_id, "")

            except Exception as e:
                if attempt < self.retry_config.max_attempts - 1:
                    delay = random.uniform(
                        self.retry_config.min_delay_seconds,
                        self.retry_config.max_delay_seconds
                    )
                    print(f"\n  ⚠ Attempt {attempt + 1} failed, retrying in {delay:.1f}s...")
                    print(f"     Error: {type(e).__name__}: {str(e)}")
                    time.sleep(delay)
                else:
                    # Last attempt failed - print full traceback and return placeholder
                    print(f"\n  ✗ All {self.retry_config.max_attempts} attempts failed for sample {p2_sample.sample_id}")
                    print("=" * 80)
                    traceback.print_exc()
                    print("=" * 80)

                    from dsl_model.nodes import NoteNodeData
                    empty_data: NodeData = NoteNodeData(
                        title="Generation Failed",
                        theme="yellow",
                        text=f"Failed to generate {p2_sample.node_type} node"
                    )
                    # Get full error message with traceback
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    return (empty_data, "", error_msg)

        # This should never be reached, but add for type checker
        from dsl_model.nodes import NoteNodeData
        empty_data_final: NodeData = NoteNodeData(
            title="Generation Failed",
            theme="yellow",
            text="Max retries exceeded"
        )
        return (empty_data_final, "", "Max retries exceeded")
