"""
Configuration loader for evaluation system.
Supports YAML config with local overrides and environment variable substitution.
"""

import os
from pathlib import Path
from typing import Any, List, Dict
import yaml
from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """API configuration"""
    anthropic_api_key: str


class ModelsConfig(BaseModel):
    """Global models configuration"""
    generation: str = Field(description="Node generation model")
    inference: str = Field(description="User message inference model")
    judge: str = Field(description="Judge evaluation model")


class DatasetConfig(BaseModel):
    """Dataset sampling configuration"""
    source_dsl_dir: str
    total_samples: int
    max_samples_per_file: int
    excluded_node_types: List[str]


class UserMessageInferenceConfig(BaseModel):
    """User message inference configuration"""
    prompt_template: str
    temperature: float
    max_tokens: int


class JudgeConfig(BaseModel):
    """Judge evaluation configuration"""
    prompt_template: str
    temperature: float
    max_tokens: int


class RetryConfig(BaseModel):
    """Retry configuration for API calls"""
    max_attempts: int = Field(ge=1, le=10)
    min_delay_seconds: float = Field(ge=0.0)
    max_delay_seconds: float = Field(ge=0.0)


class OutputConfig(BaseModel):
    """Output configuration"""
    judge_results_json: str
    analysis_report: str


class EvaluationConfig(BaseModel):
    """Evaluation system configuration"""
    dataset: DatasetConfig
    user_message_inference: UserMessageInferenceConfig
    judge: JudgeConfig
    retry: RetryConfig
    output: OutputConfig


class GlobalConfig(BaseModel):
    """Complete configuration"""
    api: APIConfig
    models: ModelsConfig
    evaluation: EvaluationConfig


class ConfigLoader:
    """Configuration loader with support for local overrides and env vars"""

    @staticmethod
    def load(
        config_file: str = "config.yml",
        local_config_file: str = "local.config.yml"
    ) -> GlobalConfig:
        """
        Load configuration with local overrides and environment variable substitution.

        Args:
            config_file: Default config file path
            local_config_file: Local config file path (optional)

        Returns:
            GlobalConfig object
        """
        # Load default config
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Merge local config if exists
        if Path(local_config_file).exists():
            with open(local_config_file, 'r', encoding='utf-8') as f:
                local_data = yaml.safe_load(f)
            if local_data and config_data:
                config_data = ConfigLoader._deep_merge(config_data, local_data)

        # Resolve environment variables
        if config_data:
            config_data = ConfigLoader._resolve_env_vars(config_data)
        else:
            raise ValueError("Config data is None")

        return GlobalConfig.model_validate(config_data)

    @staticmethod
    def _deep_merge(base: Any, override: Any) -> Any:
        """Deep merge two dictionaries"""
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override

        result: Dict[str, Any] = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _resolve_env_vars(data: Any) -> Any:
        """Recursively resolve environment variables in config"""
        if isinstance(data, dict):
            return {k: ConfigLoader._resolve_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [ConfigLoader._resolve_env_vars(v) for v in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            var_name = data[2:-1]
            return os.getenv(var_name, data)
        else:
            return data
