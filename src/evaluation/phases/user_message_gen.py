"""
Phase 2: User Message Generator
Infers user intent from removed node configuration using AI.
"""

import time
import random
import json
import traceback
from typing import List
from concurrent.futures import as_completed
from anthropic import Anthropic
from ai_workflow_action import DifyWorkflowDslFile, DifyWorkflowContextBuilder
from ai_workflow_action.parallel_service import ParallelService
from ..models import Phase1Dataset, Phase1Sample, Phase2Dataset, Phase2Sample
from ai_workflow_action.config_loader import ConfigLoader


class UserMessageGenerator:
    """User message generator with retry mechanism"""

    def __init__(self):
        config = ConfigLoader.get_config()
        self.model = config.models.user_intent_inference
        self.config = config.evaluation.user_message_inference
        self.retry_config = config.evaluation.retry
        self.client = Anthropic(api_key=config.api.anthropic_api_key)

    def generate(self, phase1_data: Phase1Dataset) -> Phase2Dataset:
        """
        Generate user messages for all samples.

        Args:
            phase1_data: Phase 1 dataset

        Returns:
            Phase 2 dataset with user_message field populated
        """
        # Submit all tasks to thread pool
        futures = {
            ParallelService.submit(self._generate_with_retry_safe, sample): (i, sample)
            for i, sample in enumerate(phase1_data.samples)
        }

        # Collect results as they complete
        results: List[tuple[int, Phase1Sample, str]] = []
        for future in as_completed(futures):
            i, sample = futures[future]
            user_message = future.result()
            results.append((i, sample, user_message))
            status = "✓" if user_message else "✗"
            print(f"  [{len(results)}/{len(phase1_data.samples)}] Sample {sample.sample_id} {status}")

        # Sort by original order
        results.sort(key=lambda x: x[0])

        # Create Phase2 samples (keep all samples, even if user_message is empty)
        phase2_samples = [
            Phase2Sample(
                sample_id=sample.sample_id,
                source_file=sample.source_file,
                masked_workflow=sample.masked_workflow,
                node_type=sample.node_type,
                after_node_id=sample.after_node_id,
                app_name=sample.app_name,
                app_description=sample.app_description,
                user_message=user_message or "[GENERATION FAILED]"
            )
            for _, sample, user_message in results
        ]

        return Phase2Dataset(samples=phase2_samples, metadata=phase1_data.metadata)

    def _generate_with_retry_safe(self, p1_sample: Phase1Sample) -> str:
        """Safe wrapper that catches all exceptions and returns empty string on failure"""
        try:
            return self._generate_with_retry(p1_sample)
        except Exception as e:
            # Capture full error with stacktrace
            error_msg = f"[Phase 2] {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            p1_sample.errors.append(error_msg)
            # Return placeholder
            return "[GENERATION FAILED]"

    def _generate_with_retry(self, p1_sample: Phase1Sample) -> str:
        """Generate with retry mechanism"""
        for attempt in range(self.retry_config.max_attempts):
            try:
                prompt = self._build_prompt(p1_sample)

                response = self.client.messages.create(
                    model=self.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )

                return response.content[0].text.strip()

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
                    print(f"\n  ✗ All {self.retry_config.max_attempts} attempts failed")
                    print("=" * 80)
                    traceback.print_exc()
                    print("=" * 80)
                    raise e

        # This should never be reached, but add for type checker
        raise RuntimeError("Max retries exceeded")

    def _build_prompt(self, p1_sample: Phase1Sample) -> str:
        """Build inference prompt"""
        # Load template
        with open(self.config.prompt_template, 'r', encoding='utf-8') as f:
            template = f.read()

        # Extract workflow info from memory
        temp_wf = DifyWorkflowDslFile.__new__(DifyWorkflowDslFile)
        temp_wf.dsl = p1_sample.masked_workflow.dsl
        temp_wf.file_path = None

        # Get workflow summary
        nodes = DifyWorkflowContextBuilder.extract_topological_sequence(temp_wf)
        workflow_summary = '\n'.join(
            f"- [{n.data.type}] {n.data.title}" for n in nodes
        )

        # Extract key config
        key_config = self._extract_key_config(p1_sample)

        # Fill template
        return template.format(
            app_name=p1_sample.app_name,
            app_description=p1_sample.app_description,
            workflow_summary=workflow_summary,
            node_type=p1_sample.node_type,
            node_title=p1_sample.masked_workflow.removed_node_data.title,
            key_config=key_config,
            after_node_id=p1_sample.after_node_id
        )

    def _extract_key_config(self, p1_sample: Phase1Sample) -> str:
        """Extract key configuration based on node type"""
        node_data = p1_sample.masked_workflow.removed_node_data
        node_type = p1_sample.node_type

        if node_type == "llm":
            from dsl_model.nodes import LLMNodeData
            if isinstance(node_data, LLMNodeData):
                model_info = node_data.model
                system_msg = next(
                    (p.text[:200] for p in node_data.prompt_template if p.role == "system"),
                    ""
                )
                return f"Model: {model_info.provider}/{model_info.name}\nSystem: {system_msg}..."

        elif node_type == "code":
            from dsl_model.nodes import CodeNodeData
            if isinstance(node_data, CodeNodeData):
                return f"Code:\n{node_data.code[:300]}..."

        elif node_type == "http-request":
            from dsl_model.nodes import HTTPRequestNodeData
            if isinstance(node_data, HTTPRequestNodeData):
                return f"Method: {node_data.method}\nURL: {node_data.url}"

        # Default: return JSON dump
        return json.dumps(node_data.model_dump(), ensure_ascii=False)[:300]
