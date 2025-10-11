"""
Evaluation Pipeline
Orchestrates all 6 phases of the evaluation process.
"""

from .models import (
    Phase1Dataset, Phase2Dataset, Phase3Dataset,
    Phase4Dataset, Phase5Dataset, EvaluationResults
)
from ai_workflow_action.config_loader import GlobalConfig


class EvaluationPipeline:
    """Evaluation pipeline (6 phases, in-memory execution)"""

    def __init__(self, config: GlobalConfig):
        self.config = config

    def run(self) -> EvaluationResults:
        """
        Run complete evaluation pipeline.

        Returns:
            Final evaluation results
        """
        print("\n=== Evaluation Pipeline Started ===\n")

        # Phase 1: Build dataset
        phase1_output: Phase1Dataset = self._run_phase1()

        # Phase 2: Infer user messages
        phase2_output: Phase2Dataset = self._run_phase2(phase1_output)

        # Phase 3: Generate nodes (reuses AiWorkflowAction API)
        phase3_output: Phase3Dataset = self._run_phase3(phase2_output)

        # Phase 4: Validate workflows
        phase4_output: Phase4Dataset = self._run_phase4(phase3_output)

        # Phase 5: Judge evaluation
        phase5_output: Phase5Dataset = self._run_phase5(phase4_output)

        # Phase 6: Analyze and report
        results: EvaluationResults = self._run_phase6(phase5_output)

        print("\n=== Evaluation Completed ===\n")
        return results

    def _run_phase1(self) -> Phase1Dataset:
        """Phase 1: Build dataset"""
        print("[Phase 1/6] Building dataset...")
        from .phases.dataset_builder import DatasetBuilder

        builder = DatasetBuilder(self.config.evaluation.dataset)
        dataset = builder.build_dataset()

        print(f"  ✓ Sampled {len(dataset.samples)} nodes")
        return dataset

    def _run_phase2(self, phase1_output: Phase1Dataset) -> Phase2Dataset:
        """Phase 2: Generate user messages"""
        print("\n[Phase 2/6] Inferring user messages...")
        from .phases.user_message_gen import UserMessageGenerator

        generator = UserMessageGenerator(
            self.config.models.inference,
            self.config.evaluation.user_message_inference,
            self.config.evaluation.retry
        )
        dataset = generator.generate(phase1_output)

        print(f"  ✓ Generated {len(dataset.samples)} messages")
        return dataset

    def _run_phase3(self, phase2_output: Phase2Dataset) -> Phase3Dataset:
        """Phase 3: Generate nodes"""
        print("\n[Phase 3/6] Generating nodes (using AiWorkflowAction API)...")
        from .phases.node_generator import NodeGenerator

        generator = NodeGenerator(
            self.config.api.anthropic_api_key,
            self.config.models.generation,
            self.config.evaluation.retry
        )
        dataset = generator.generate(phase2_output)

        success_count = sum(1 for s in dataset.samples if not s.generation_error)
        print(f"  ✓ Generated {success_count}/{len(dataset.samples)} nodes")
        return dataset

    def _run_phase4(self, phase3_output: Phase3Dataset) -> Phase4Dataset:
        """Phase 4: Validate workflows"""
        print("\n[Phase 4/6] Validating workflows...")
        from .phases.workflow_validator import WorkflowValidator

        validator = WorkflowValidator()
        dataset = validator.validate(phase3_output)

        success_count = sum(1 for s in dataset.samples if s.validation_success)
        print(f"  ✓ Validated {success_count}/{len(dataset.samples)} workflows")
        return dataset

    def _run_phase5(self, phase4_output: Phase4Dataset) -> Phase5Dataset:
        """Phase 5: Judge evaluation"""
        print("\n[Phase 5/6] Evaluating with LLM Judge...")
        from .phases.llm_judge import LLMJudge

        judge = LLMJudge(
            self.config.models.judge,
            self.config.evaluation.judge,
            self.config.evaluation.retry
        )
        dataset = judge.evaluate(phase4_output)

        avg_score = sum(s.final_score for s in dataset.samples) / len(dataset.samples)
        print(f"  ✓ Evaluated {len(dataset.samples)} samples (Avg: {avg_score:.2f})")
        return dataset

    def _run_phase6(self, phase5_output: Phase5Dataset) -> EvaluationResults:
        """Phase 6: Analyze and generate report"""
        print("\n[Phase 6/6] Analyzing and generating report...")
        from .phases.analyzer import Analyzer

        analyzer = Analyzer(self.config.evaluation.output)
        results = analyzer.analyze(phase5_output, self.config)

        # Save outputs
        results.save_json(self.config.evaluation.output.judge_results_json)
        analyzer.generate_report(results)

        print(f"  ✓ Saved: {self.config.evaluation.output.judge_results_json}")
        print(f"  ✓ Saved: {self.config.evaluation.output.analysis_report}")

        return results
