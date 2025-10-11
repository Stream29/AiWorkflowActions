"""
Phase 2: User Message Generator
Infers user intent from removed node configuration using AI.
"""

import time
import random
import json
from typing import List
from anthropic import Anthropic
from ai_workflow_action import DifyWorkflowDslFile, DifyWorkflowContextBuilder
from ..models import Phase1Dataset, Phase1Sample, Phase2Dataset, Phase2Sample
from ai_workflow_action.config_loader import ConfigLoader


class UserMessageGenerator:
    """User message generator with retry mechanism"""

    def __init__(self):
        config = ConfigLoader.get_config()
        self.model = config.models.inference
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
        phase2_samples: List[Phase2Sample] = []

        for i, p1_sample in enumerate(phase1_data.samples):
            print(f"  [{i+1}/{len(phase1_data.samples)}] Sample {p1_sample.sample_id}...", end=" ")

            user_message = self._generate_with_retry(p1_sample)

            phase2_sample = Phase2Sample(
                sample_id=p1_sample.sample_id,
                source_file=p1_sample.source_file,
                masked_workflow=p1_sample.masked_workflow,
                node_type=p1_sample.node_type,
                after_node_id=p1_sample.after_node_id,
                app_name=p1_sample.app_name,
                app_description=p1_sample.app_description,
                user_message=user_message
            )
            phase2_samples.append(phase2_sample)

            print("✓")

        return Phase2Dataset(samples=phase2_samples, metadata=phase1_data.metadata)

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
                    time.sleep(delay)
                else:
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
