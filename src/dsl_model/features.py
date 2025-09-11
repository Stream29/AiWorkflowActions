from typing import Any, Dict, List

from pydantic import BaseModel, Field, ConfigDict

from .graph import Graph


class FileUploadConfig(BaseModel):
    """File upload configuration"""
    audio_file_size_limit: int = Field(default=50, ge=1)
    batch_count_limit: int = Field(default=5, ge=1)
    file_size_limit: int = Field(default=15, ge=1)
    image_file_size_limit: int = Field(default=10, ge=1)
    video_file_size_limit: int = Field(default=100, ge=1)
    workflow_file_upload_limit: int = Field(default=10, ge=1)


class FileUploadFeature(BaseModel):
    """File upload feature configuration"""
    enabled: bool = Field(default=False)
    allowed_file_extensions: List[str] = Field(default_factory=list)
    allowed_file_types: List[str] = Field(default_factory=list)
    allowed_file_upload_methods: List[str] = Field(
        default_factory=lambda: ["local_file", "remote_url"]
    )
    fileUploadConfig: FileUploadConfig = Field(default_factory=FileUploadConfig)
    number_limits: int = Field(default=3, ge=1)


class WorkflowFeatures(BaseModel):
    """Workflow features configuration"""
    file_upload: FileUploadFeature = Field(default_factory=FileUploadFeature)
    opening_statement: str = Field(default="")
    retriever_resource: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True}
    )
    sensitive_word_avoidance: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False}
    )
    speech_to_text: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False}
    )
    suggested_questions: List[str] = Field(default_factory=list)
    suggested_questions_after_answer: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False}
    )
    text_to_speech: Dict[str, Any] = Field(
        default_factory=lambda: {"enabled": False, "language": "", "voice": ""}
    )

    model_config = ConfigDict(extra="allow")


class Workflow(BaseModel):
    """Complete workflow definition"""
    conversation_variables: List[Any] = Field(default_factory=list)
    environment_variables: List[Any] = Field(default_factory=list)
    features: WorkflowFeatures = Field(default_factory=WorkflowFeatures)
    graph: Graph
