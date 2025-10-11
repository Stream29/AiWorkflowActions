"""
Phase 6: Analyzer
Analyzes evaluation results and generates report.
"""

from datetime import datetime
from typing import List, Dict
from ..models import (
    Phase5Dataset, Phase5Sample, EvaluationResults, SingleSampleResult,
    OverallStats, ScoreDistribution, NodeTypeScore, ConfigSummary
)
from ai_workflow_action.config_loader import ConfigLoader


class Analyzer:
    """Analyzer for evaluation results"""

    def __init__(self):
        config = ConfigLoader.get_config()
        self.config = config.evaluation.output

    def analyze(self, phase5_data: Phase5Dataset) -> EvaluationResults:
        """
        Analyze evaluation results.

        Args:
            phase5_data: Phase 5 dataset

        Returns:
            Evaluation results
        """
        global_config = ConfigLoader.get_config()

        # Convert to simplified results
        sample_results: List[SingleSampleResult] = []
        for p5_sample in phase5_data.samples:
            sample_results.append(SingleSampleResult(
                sample_id=p5_sample.sample_id,
                node_type=p5_sample.node_type,
                source_file=p5_sample.source_file,
                user_message=p5_sample.user_message,
                final_score=p5_sample.final_score,
                variable_analysis=p5_sample.variable_analysis,
                structure_analysis=p5_sample.structure_analysis,
                semantic_quality=p5_sample.semantic_quality,
                config_reasonableness=p5_sample.config_reasonableness,
                summary=p5_sample.judge_summary
            ))

        # Compute statistics
        overall_stats = self._compute_stats(phase5_data.samples)

        # Create config summary
        config_summary = ConfigSummary(
            generation_model=global_config.models.generation,
            inference_model=global_config.models.inference,
            judge_model=global_config.models.judge,
            total_samples_target=global_config.evaluation.dataset.total_samples
        )

        # Compute total samples
        total_samples = sum(
            d.count for d in phase5_data.metadata.node_type_distributions
        )

        return EvaluationResults(
            config_summary=config_summary,
            total_samples=total_samples,
            evaluated_samples=len(sample_results),
            sample_results=sample_results,
            overall_stats=overall_stats
        )

    def generate_report(self, results: EvaluationResults) -> None:
        """Generate Markdown analysis report"""
        stats = results.overall_stats
        dist = stats.score_distribution

        report = f"""# Evaluation Analysis Report

**Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Generation Model**: {results.config_summary.generation_model}
**Judge Model**: {results.config_summary.judge_model}

## 1. Overall Statistics

- Total Samples: {results.total_samples}
- Evaluated Samples: {results.evaluated_samples}
- Average Score: {stats.avg_score:.2f} / 7.0
- Average Jaccard: {stats.avg_jaccard:.3f}
- Low Score Count (< 4.0): {stats.low_score_count}

## 2. Score Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 6.5-7.0 | {dist.range_6_5_to_7_0} | {(dist.range_6_5_to_7_0 / results.evaluated_samples * 100):.1f}% |
| 5.5-6.5 | {dist.range_5_5_to_6_5} | {(dist.range_5_5_to_6_5 / results.evaluated_samples * 100):.1f}% |
| 4.0-5.5 | {dist.range_4_0_to_5_5} | {(dist.range_4_0_to_5_5 / results.evaluated_samples * 100):.1f}% |
| 2.5-4.0 | {dist.range_2_5_to_4_0} | {(dist.range_2_5_to_4_0 / results.evaluated_samples * 100):.1f}% |
| 1.0-2.5 | {dist.range_1_0_to_2_5} | {(dist.range_1_0_to_2_5 / results.evaluated_samples * 100):.1f}% |

## 3. By Node Type

| Type | Average Score | Sample Count |
|------|---------------|--------------|
"""
        for node_type_score in sorted(stats.node_type_scores, key=lambda x: x.avg_score):
            report += f"| {node_type_score.node_type} | {node_type_score.avg_score:.2f} | {node_type_score.sample_count} |\n"

        with open(self.config.analysis_report, 'w', encoding='utf-8') as f:
            f.write(report)

    def _compute_stats(self, samples: List[Phase5Sample]) -> OverallStats:
        """Compute overall statistics"""
        scores = [s.final_score for s in samples]

        # Score distribution
        score_dist = ScoreDistribution(
            **{
                "6.5-7.0": sum(1 for s in scores if s >= 6.5),
                "5.5-6.5": sum(1 for s in scores if 5.5 <= s < 6.5),
                "4.0-5.5": sum(1 for s in scores if 4.0 <= s < 5.5),
                "2.5-4.0": sum(1 for s in scores if 2.5 <= s < 4.0),
                "1.0-2.5": sum(1 for s in scores if s < 2.5),
            }
        )

        # Node type scores
        type_scores_map: Dict[str, List[float]] = {}
        for sample in samples:
            if sample.node_type not in type_scores_map:
                type_scores_map[sample.node_type] = []
            type_scores_map[sample.node_type].append(sample.final_score)

        node_type_scores = [
            NodeTypeScore(
                node_type=node_type,
                avg_score=sum(scores_list) / len(scores_list),
                sample_count=len(scores_list)
            )
            for node_type, scores_list in type_scores_map.items()
        ]

        # Overall metrics
        avg_jaccard = sum(
            s.variable_analysis.jaccard_similarity for s in samples
        ) / len(samples)

        return OverallStats(
            avg_score=sum(scores) / len(scores),
            avg_jaccard=avg_jaccard,
            score_distribution=score_dist,
            node_type_scores=node_type_scores,
            low_score_count=sum(1 for s in scores if s < 4.0)
        )
