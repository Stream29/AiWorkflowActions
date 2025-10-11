"""
Data models for the evaluation system.
Strict typing with proper generic types from typing module.
"""

from typing import List, Dict
from pydantic import BaseModel, Field, ConfigDict
from dsl_model import DifyWorkflowDSL, NodeData


# ============================================================================
# Metadata Models
# ============================================================================

class NodeTypeDistribution(BaseModel):
    """Node type distribution statistics"""
    node_type: str
    count: int


class DatasetMetadata(BaseModel):
    """Metadata for dataset"""
    source_dir: str = Field(description="Source DSL directory")
    total_files_scanned: int = Field(default=0, description="Total files scanned")
    total_removable_nodes: int = Field(default=0, description="Total removable nodes found")
    node_type_distributions: List[NodeTypeDistribution] = Field(default_factory=list)


# ============================================================================
# Masked Workflow
# ============================================================================

class MaskedWorkflow(BaseModel):
    """Internal memory-only masked workflow"""
    dsl: DifyWorkflowDSL = Field(description="DSL object")
    removed_node_id: str = Field(description="ID of removed node")
    removed_node_data: NodeData = Field(description="Data of removed node")

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Phase 1: Dataset Builder
# ============================================================================

class Phase1Sample(BaseModel):
    """Phase 1 sample data"""
    sample_id: int
    source_file: str
    masked_workflow: MaskedWorkflow
    node_type: str
    after_node_id: str
    app_name: str
    app_description: str

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Phase1Dataset(BaseModel):
    """Phase 1 dataset"""
    samples: List[Phase1Sample]
    metadata: DatasetMetadata

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Phase 2: User Message Generator
# ============================================================================

class Phase2Sample(Phase1Sample):
    """Phase 2 sample data (adds user_message)"""
    user_message: str


class Phase2Dataset(BaseModel):
    """Phase 2 dataset"""
    samples: List[Phase2Sample]
    metadata: DatasetMetadata

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Phase 3: Node Generator
# ============================================================================

class Phase3Sample(Phase2Sample):
    """Phase 3 sample data (adds generated output)"""
    actual_output: NodeData
    generated_node_id: str
    generation_error: str = Field(default="")


class Phase3Dataset(BaseModel):
    """Phase 3 dataset"""
    samples: List[Phase3Sample]
    metadata: DatasetMetadata

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Phase 4: Workflow Validator
# ============================================================================

class Phase4Sample(Phase3Sample):
    """Phase 4 sample data (adds validation result)"""
    validation_success: bool
    validation_error: str = Field(default="")


class Phase4Dataset(BaseModel):
    """Phase 4 dataset"""
    samples: List[Phase4Sample]
    metadata: DatasetMetadata

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Phase 5: LLM Judge
# ============================================================================

class VariableAnalysis(BaseModel):
    """Variable reference analysis"""
    expected_variables: List[str]
    actual_variables: List[str]
    jaccard_similarity: float
    missing_variables: List[str]
    extra_variables: List[str]


class StructureAnalysis(BaseModel):
    """Structure completeness analysis"""
    required_fields_present: int
    total_required_fields: int
    completeness_ratio: float
    missing_fields: List[str]


class Phase5Sample(Phase4Sample):
    """Phase 5 sample data (adds evaluation result)"""
    variable_analysis: VariableAnalysis
    structure_analysis: StructureAnalysis
    semantic_quality: str
    config_reasonableness: str
    final_score: float = Field(ge=1.0, le=7.0)
    judge_summary: str


class Phase5Dataset(BaseModel):
    """Phase 5 dataset"""
    samples: List[Phase5Sample]
    metadata: DatasetMetadata

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Final Output Models
# ============================================================================

class SingleSampleResult(BaseModel):
    """Single sample evaluation result (simplified for output)"""
    sample_id: int
    node_type: str
    source_file: str
    user_message: str
    final_score: float
    variable_analysis: VariableAnalysis
    structure_analysis: StructureAnalysis
    semantic_quality: str
    config_reasonableness: str
    summary: str


class ScoreDistribution(BaseModel):
    """Score distribution statistics"""
    range_6_5_to_7_0: int = Field(default=0, alias="6.5-7.0")
    range_5_5_to_6_5: int = Field(default=0, alias="5.5-6.5")
    range_4_0_to_5_5: int = Field(default=0, alias="4.0-5.5")
    range_2_5_to_4_0: int = Field(default=0, alias="2.5-4.0")
    range_1_0_to_2_5: int = Field(default=0, alias="1.0-2.5")

    model_config = ConfigDict(populate_by_name=True)


class NodeTypeScore(BaseModel):
    """Score for a specific node type"""
    node_type: str
    avg_score: float
    sample_count: int


class OverallStats(BaseModel):
    """Overall evaluation statistics"""
    avg_score: float
    avg_jaccard: float
    score_distribution: ScoreDistribution
    node_type_scores: List[NodeTypeScore]
    low_score_count: int = Field(description="Count of samples with score < 4.0")


class ConfigSummary(BaseModel):
    """Configuration summary for the evaluation"""
    generation_model: str
    inference_model: str
    judge_model: str
    total_samples_target: int


class EvaluationResults(BaseModel):
    """Final evaluation results"""
    version: str = Field(default="1.0")
    config_summary: ConfigSummary
    total_samples: int
    evaluated_samples: int
    sample_results: List[SingleSampleResult]
    overall_stats: OverallStats

    def save_json(self, file_path: str) -> None:
        """Save to JSON file"""
        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(by_alias=True), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, file_path: str) -> "EvaluationResults":
        """Load from JSON file"""
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.model_validate(data)
