"""
Phase 4: Workflow Validator
Validates that generated nodes can be inserted into workflows.
"""

import traceback
from typing import List
from ai_workflow_action import DifyWorkflowDslFile
from ai_workflow_action.node_type_util import NodeTypeUtil
from dsl_model import NodeType
from ..models import Phase3Dataset, Phase3Sample, Phase4Dataset, Phase4Sample


class WorkflowValidator:
    """Workflow validator (in-memory validation)"""

    def validate(self, phase3_data: Phase3Dataset) -> Phase4Dataset:
        """
        Validate that generated nodes can be inserted.

        Args:
            phase3_data: Phase 3 dataset

        Returns:
            Phase 4 dataset with validation_success populated
        """
        phase4_samples: List[Phase4Sample] = []

        for p3_sample in phase3_data.samples:
            # Skip validation if there were errors in previous phases
            if p3_sample.errors:
                # Previous phase failed, skip validation
                phase4_sample = Phase4Sample(
                    sample_id=p3_sample.sample_id,
                    source_file=p3_sample.source_file,
                    masked_workflow=p3_sample.masked_workflow,
                    node_type=p3_sample.node_type,
                    after_node_id=p3_sample.after_node_id,
                    app_name=p3_sample.app_name,
                    app_description=p3_sample.app_description,
                    user_message=p3_sample.user_message,
                    errors=p3_sample.errors,
                    actual_output=p3_sample.actual_output,
                    generated_node_id=p3_sample.generated_node_id,
                    validation_success=False
                )
                phase4_samples.append(phase4_sample)
                continue

            try:
                # Validate schema
                node_model = NodeTypeUtil.get_node_data_model(NodeType(p3_sample.node_type))
                node_data = node_model.model_validate(p3_sample.actual_output.model_dump())

                # Validate insertion
                temp_wf = DifyWorkflowDslFile.__new__(DifyWorkflowDslFile)
                temp_wf.dsl = p3_sample.masked_workflow.dsl
                temp_wf.file_path = None

                temp_wf.add_node_after(p3_sample.after_node_id, node_data)

                # Validate DSL integrity
                temp_wf.dsl.model_validate(temp_wf.dsl.model_dump())

                # Success
                phase4_sample = Phase4Sample(
                    sample_id=p3_sample.sample_id,
                    source_file=p3_sample.source_file,
                    masked_workflow=p3_sample.masked_workflow,
                    node_type=p3_sample.node_type,
                    after_node_id=p3_sample.after_node_id,
                    app_name=p3_sample.app_name,
                    app_description=p3_sample.app_description,
                    user_message=p3_sample.user_message,
                    errors=p3_sample.errors,
                    actual_output=p3_sample.actual_output,
                    generated_node_id=p3_sample.generated_node_id,
                    validation_success=True
                )

            except Exception as e:
                # Validation failed - print full traceback and append error
                print(f"\n  ✗ Validation failed for sample {p3_sample.sample_id}")
                print("=" * 80)
                traceback.print_exc()
                print("=" * 80)

                # Capture full error with stacktrace
                error_msg = f"[Phase 4] {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                p3_sample.errors.append(error_msg)

                phase4_sample = Phase4Sample(
                    sample_id=p3_sample.sample_id,
                    source_file=p3_sample.source_file,
                    masked_workflow=p3_sample.masked_workflow,
                    node_type=p3_sample.node_type,
                    after_node_id=p3_sample.after_node_id,
                    app_name=p3_sample.app_name,
                    app_description=p3_sample.app_description,
                    user_message=p3_sample.user_message,
                    errors=p3_sample.errors,
                    actual_output=p3_sample.actual_output,
                    generated_node_id=p3_sample.generated_node_id,
                    validation_success=False
                )

            phase4_samples.append(phase4_sample)

        return Phase4Dataset(samples=phase4_samples, metadata=phase3_data.metadata)
