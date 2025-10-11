"""
Phase 5: LLM Judge
Evaluates generated nodes using LLM Judge with structured output.
"""

import time
import random
import json
import traceback
from typing import TypedDict, List, Dict, Any
from anthropic import Anthropic
from ..models import (
    Phase4Dataset, Phase4Sample, Phase5Dataset, Phase5Sample,
    VariableAnalysis, StructureAnalysis
)
from ai_workflow_action.config_loader import ConfigLoader


class JudgeResultDict(TypedDict):
    """Type for judge result dictionary"""
    variable_analysis: Dict[str, Any]
    structure_analysis: Dict[str, Any]
    semantic_quality: str
    config_reasonableness: str
    final_score: float
    summary: str


class LLMJudge:
    """LLM Judge evaluator with retry mechanism"""

    def __init__(self):
        config = ConfigLoader.get_config()
        self.model = config.models.judge
        self.config = config.evaluation.judge
        self.retry_config = config.evaluation.retry
        self.client = Anthropic(api_key=config.api.anthropic_api_key)

    def evaluate(self, phase4_data: Phase4Dataset) -> Phase5Dataset:
        """
        Evaluate samples using LLM Judge.

        Args:
            phase4_data: Phase 4 dataset

        Returns:
            Phase 5 dataset with evaluation results
        """
        phase5_samples: List[Phase5Sample] = []

        for i, p4_sample in enumerate(phase4_data.samples):
            if not p4_sample.validation_success:
                print(f"  [{i+1}/{len(phase4_data.samples)}] Skip sample {p4_sample.sample_id}")
                continue

            print(f"  [{i+1}/{len(phase4_data.samples)}] Sample {p4_sample.sample_id}...", end=" ")

            result = self._evaluate_with_retry(p4_sample)

            var_analysis_data = result["variable_analysis"]
            struct_analysis_data = result["structure_analysis"]

            phase5_sample = Phase5Sample(
                sample_id=p4_sample.sample_id,
                source_file=p4_sample.source_file,
                masked_workflow=p4_sample.masked_workflow,
                node_type=p4_sample.node_type,
                after_node_id=p4_sample.after_node_id,
                app_name=p4_sample.app_name,
                app_description=p4_sample.app_description,
                user_message=p4_sample.user_message,
                actual_output=p4_sample.actual_output,
                generated_node_id=p4_sample.generated_node_id,
                generation_error=p4_sample.generation_error,
                validation_success=p4_sample.validation_success,
                validation_error=p4_sample.validation_error,
                variable_analysis=VariableAnalysis(
                    expected_variables=var_analysis_data["expected_variables"],
                    actual_variables=var_analysis_data["actual_variables"],
                    jaccard_similarity=var_analysis_data["jaccard_similarity"],
                    missing_variables=var_analysis_data["missing_variables"],
                    extra_variables=var_analysis_data["extra_variables"]
                ),
                structure_analysis=StructureAnalysis(
                    required_fields_present=struct_analysis_data["required_fields_present"],
                    total_required_fields=struct_analysis_data["total_required_fields"],
                    completeness_ratio=struct_analysis_data["completeness_ratio"],
                    missing_fields=struct_analysis_data["missing_fields"]
                ),
                semantic_quality=str(result["semantic_quality"]),
                config_reasonableness=str(result["config_reasonableness"]),
                final_score=float(result["final_score"]),
                judge_summary=str(result["summary"])
            )
            phase5_samples.append(phase5_sample)

            print(f"✓ ({result['final_score']:.1f})")

        return Phase5Dataset(samples=phase5_samples, metadata=phase4_data.metadata)

    def _evaluate_with_retry(self, p4_sample: Phase4Sample) -> JudgeResultDict:
        """Evaluate with retry mechanism"""
        for attempt in range(self.retry_config.max_attempts):
            try:
                prompt = self._build_prompt(p4_sample)

                response = self.client.messages.create(
                    model=self.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )

                return self._parse_json(response.content[0].text)

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
                    print(f"\n  ✗ All {self.retry_config.max_attempts} attempts failed for sample {p4_sample.sample_id}")
                    print("=" * 80)
                    traceback.print_exc()
                    print("=" * 80)
                    raise e

        # This should never be reached, but add for type checker
        raise RuntimeError("Max retries exceeded")

    def _build_prompt(self, p4_sample: Phase4Sample) -> str:
        """Build judge evaluation prompt"""
        with open(self.config.prompt_template, 'r', encoding='utf-8') as f:
            template = f.read()

        expected_json = json.dumps(
            p4_sample.masked_workflow.removed_node_data.model_dump(),
            indent=2,
            ensure_ascii=False
        )

        actual_json = json.dumps(
            p4_sample.actual_output.model_dump(),
            indent=2,
            ensure_ascii=False
        )

        return template + f"""

## Expected Output
```json
{expected_json}
```

## Actual Output
```json
{actual_json}
```
"""

    def _parse_json(self, text: str) -> JudgeResultDict:
        """Parse JSON from LLM response"""
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        return json.loads(cleaned)
