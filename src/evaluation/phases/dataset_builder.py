"""
Phase 1: Dataset Builder
Samples nodes from DSL files and creates masked workflows in memory.
"""

import random
import copy
import traceback
from pathlib import Path
from typing import List, Tuple, Dict

from dsl_model import Node
from ai_workflow_action import DifyWorkflowDslFile
from ..models import Phase1Sample, Phase1Dataset, MaskedWorkflow, DatasetMetadata, NodeTypeDistribution
from ai_workflow_action.config_loader import ConfigLoader


class DatasetBuilder:
    """Dataset builder for evaluation system"""

    def __init__(self, verbose: bool = True):
        config = ConfigLoader.get_config()
        self.config = config.evaluation.dataset
        self.verbose = verbose

    def build_dataset(self) -> Phase1Dataset:
        """
        Build evaluation dataset by sampling nodes from DSL files.

        Returns:
            Phase1Dataset with sampled nodes
        """
        # Scan DSL files
        dsl_files = list(Path(self.config.source_dsl_dir).rglob('*.yml'))
        dsl_files.extend(list(Path(self.config.source_dsl_dir).rglob('*.yaml')))
        print(f"  Scanning {len(dsl_files)} files...")

        # Collect candidate nodes
        candidates: List[Tuple[str, DifyWorkflowDslFile, Node]] = []
        failed_files = 0
        for dsl_file in dsl_files:
            try:
                workflow = DifyWorkflowDslFile(str(dsl_file))
                for node in workflow.dsl.workflow.graph.nodes:
                    if self._is_removable(node, workflow):
                        candidates.append((str(dsl_file), workflow, node))
            except Exception as e:
                failed_files += 1
                if self.verbose:
                    print(f"\n  ⚠ Failed to load {dsl_file}")
                    print(f"     Error: {type(e).__name__}: {str(e)}")
                continue

        if failed_files > 0:
            print(f"  ⚠ Skipped {failed_files} files due to errors")

        print(f"  Found {len(candidates)} removable nodes")

        # Random sampling with file balance
        selected = self._balanced_sample(candidates)

        # Create samples
        samples = [
            self._create_sample(wf, node, file_path, i)
            for i, (file_path, wf, node) in enumerate(selected)
        ]

        # Compute distribution
        type_counts: Dict[str, int] = {}
        for sample in samples:
            type_counts[sample.node_type] = type_counts.get(sample.node_type, 0) + 1

        distributions = [
            NodeTypeDistribution(node_type=node_type, count=count)
            for node_type, count in type_counts.items()
        ]

        metadata = DatasetMetadata(
            source_dir=self.config.source_dsl_dir,
            total_files_scanned=len(dsl_files),
            total_removable_nodes=len(candidates),
            node_type_distributions=distributions
        )

        return Phase1Dataset(samples=samples, metadata=metadata)

    def _is_removable(self, node: Node, workflow: DifyWorkflowDslFile) -> bool:
        """Check if node is removable"""
        # Exclude structural nodes
        if node.data.type in self.config.excluded_node_types:
            return False

        # Exclude nodes with no predecessors (cannot determine insertion position)
        connections = workflow.get_node_connections(node.id)
        if not connections.incoming:
            return False

        # Exclude nodes immediately after start
        for pred_id in connections.incoming:
            pred_node = workflow.get_node(pred_id)
            if pred_node and pred_node.data.type == "start":
                return False

        return True

    def _balanced_sample(
        self,
        candidates: List[Tuple[str, DifyWorkflowDslFile, Node]]
    ) -> List[Tuple[str, DifyWorkflowDslFile, Node]]:
        """Random sampling with file balance"""
        file_counts: Dict[str, int] = {}
        selected: List[Tuple[str, DifyWorkflowDslFile, Node]] = []

        random.shuffle(candidates)

        for file_path, workflow, node in candidates:
            if len(selected) >= self.config.total_samples:
                break

            if file_counts.get(file_path, 0) >= self.config.max_samples_per_file:
                continue

            selected.append((file_path, workflow, node))
            file_counts[file_path] = file_counts.get(file_path, 0) + 1

        return selected

    def _create_sample(
        self,
        workflow: DifyWorkflowDslFile,
        node: Node,
        source_file: str,
        sample_id: int
    ) -> Phase1Sample:
        """Create Phase1Sample from workflow and node"""
        # Create masked workflow (deep copy in memory)
        masked_dsl = copy.deepcopy(workflow.dsl)
        masked_dsl.workflow.graph.nodes = [
            n for n in masked_dsl.workflow.graph.nodes if n.id != node.id
        ]
        masked_dsl.workflow.graph.edges = [
            e for e in masked_dsl.workflow.graph.edges
            if e.source != node.id and e.target != node.id
        ]

        masked_wf = MaskedWorkflow(
            dsl=masked_dsl,
            removed_node_id=node.id,
            removed_node_data=node.data
        )

        # Find predecessor node
        connections = workflow.get_node_connections(node.id)
        after_node_id = connections.incoming[0] if connections.incoming else ""

        return Phase1Sample(
            sample_id=sample_id,
            source_file=source_file,
            masked_workflow=masked_wf,
            node_type=node.data.type,
            after_node_id=after_node_id,
            app_name=workflow.dsl.app.name,
            app_description=workflow.dsl.app.description
        )
