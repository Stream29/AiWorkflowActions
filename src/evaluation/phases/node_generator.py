"""
Phase 3: Node Generator
Generates nodes using AiWorkflowAction API (reuses existing code).
"""

import time
import random
import traceback
from typing import List, Tuple
from concurrent.futures import as_completed
from ai_workflow_action import DifyWorkflowDslFile, AiWorkflowAction
from ai_workflow_action.parallel_service import ParallelService
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
        # Submit all tasks to thread pool
        futures = {
            ParallelService.submit(self._generate_with_retry, sample): (i, sample)
            for i, sample in enumerate(phase2_data.samples)
        }

        # Collect results as they complete
        results: List[Tuple[int, Phase2Sample, NodeData, str]] = []
        for future in as_completed(futures):
            i, sample = futures[future]
            actual_output, generated_id = future.result()
            results.append((i, sample, actual_output, generated_id))

            status = "✓" if generated_id else "✗"
            print(f"  [{len(results)}/{len(phase2_data.samples)}] Sample {sample.sample_id} {status}")

        # Sort by original order
        results.sort(key=lambda x: x[0])

        # Create Phase3 samples (errors already in sample.errors)
        phase3_samples = [
            Phase3Sample(
                sample_id=sample.sample_id,
                source_file=sample.source_file,
                masked_workflow=sample.masked_workflow,
                node_type=sample.node_type,
                after_node_id=sample.after_node_id,
                app_name=sample.app_name,
                app_description=sample.app_description,
                user_message=sample.user_message,
                errors=sample.errors,
                actual_output=actual_output,
                generated_node_id=generated_id
            )
            for _, sample, actual_output, generated_id in results
        ]

        return Phase3Dataset(samples=phase3_samples, metadata=phase2_data.metadata)

    def _generate_with_retry(self, p2_sample: Phase2Sample) -> tuple[NodeData, str]:
        """
        Generate node with retry mechanism.

        Returns:
            Tuple of (actual_output, generated_node_id)
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

                return (generated_node.data, new_node_id)

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
                    # Last attempt failed - capture error and return placeholder
                    print(f"\n  ✗ All {self.retry_config.max_attempts} attempts failed for sample {p2_sample.sample_id}")
                    print("=" * 80)
                    traceback.print_exc()
                    print("=" * 80)

                    # Capture full error with stacktrace
                    error_msg = f"[Phase 3] {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    p2_sample.errors.append(error_msg)

                    from dsl_model.nodes import NoteNodeData
                    empty_data: NodeData = NoteNodeData(
                        title="Generation Failed",
                        theme="yellow",
                        text=f"Failed to generate {p2_sample.node_type} node"
                    )
                    return (empty_data, "")

        # This should never be reached, but add for type checker
        raise RuntimeError("Max retries exceeded")
