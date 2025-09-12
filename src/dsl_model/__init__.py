"""
Public API for the dsl_model package.

This module re-exports the most commonly used classes and enums so users can
import them directly from `dsl_model`.
"""

# Package version (fallback to project version if desired)
__version__ = "0.1.0"

# Enums
from .enums import (
    NodeType,
    VariableType,
    WorkflowMode,
    ErrorStrategy,
    LLMMode,
    CodeLanguage,
    SegmentType,
    DefaultValueType,
)

# Core building blocks
from .core import (
    VariableSelector,
    Position,
    Viewport,
    Variable,
    ModelConfig,
    PromptMessage,
    PromptConfig,
    LLMNodeChatModelMessage,
    LLMNodeCompletionModelPromptTemplate,
    ContextConfig,
    VisionConfig,
    DefaultValue,
    RetryConfig,
    BaseNodeData,
)

# Node data types and union
from .nodes import (
    StartNodeData,
    EndNodeData,
    AnswerNodeData,
    LLMNodeData,
    CodeNodeData,
    HTTPRequestNodeData,
    ToolNodeData,
    IfElseNodeData,
    TemplateTransformNodeData,
    VariableAssignerNodeData,
    KnowledgeRetrievalNodeData,
    AgentNodeData,
    IterationNodeData,
    ParameterExtractorNodeData,
    QuestionClassifierNodeData,
    IterationStartNodeData,
    LoopStartNodeData,
    LoopEndNodeData,
    VariableAggregatorNodeData,
    DocumentExtractorNodeData,
    ListOperatorNodeData,
    NoteNodeData,
    NodeData,
)

# Graph, features and app-level models
from .graph import EdgeData, Edge, Node, Graph
from .features import FileUploadConfig, FileUploadFeature, WorkflowFeatures, Workflow
from .app_models import AppMetadata, Dependency

# Root DSL model
from .dsl import DifyWorkflowDSL

__all__ = [
    # Version
    "__version__",
    # Enums
    "NodeType",
    "VariableType",
    "WorkflowMode",
    "ErrorStrategy",
    "LLMMode",
    "CodeLanguage",
    "SegmentType",
    "DefaultValueType",
    # Core
    "VariableSelector",
    "Position",
    "Viewport",
    "Variable",
    "ModelConfig",
    "PromptMessage",
    "PromptConfig",
    "LLMNodeChatModelMessage",
    "LLMNodeCompletionModelPromptTemplate",
    "ContextConfig",
    "VisionConfig",
    "DefaultValue",
    "RetryConfig",
    "BaseNodeData",
    # Nodes
    "StartNodeData",
    "EndNodeData",
    "AnswerNodeData",
    "LLMNodeData",
    "CodeNodeData",
    "HTTPRequestNodeData",
    "ToolNodeData",
    "IfElseNodeData",
    "TemplateTransformNodeData",
    "VariableAssignerNodeData",
    "KnowledgeRetrievalNodeData",
    "AgentNodeData",
    "IterationNodeData",
    "ParameterExtractorNodeData",
    "QuestionClassifierNodeData",
    "IterationStartNodeData",
    "LoopStartNodeData",
    "LoopEndNodeData",
    "VariableAggregatorNodeData",
    "DocumentExtractorNodeData",
    "ListOperatorNodeData",
    "NoteNodeData",
    "NodeData",
    # Graph / Features / App models
    "EdgeData",
    "Edge",
    "Node",
    "Graph",
    "FileUploadConfig",
    "FileUploadFeature",
    "WorkflowFeatures",
    "Workflow",
    "AppMetadata",
    "Dependency",
    # Root DSL
    "DifyWorkflowDSL",
]