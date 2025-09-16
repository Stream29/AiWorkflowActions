"""
AiWorkflowActions - AI-powered workflow node generation for Dify
Clean layered architecture:
1. DifyWorkflowDSL (from dsl_model) - Core data structures
2. DifyWorkflowDslFile - Basic operations + RAII
3. AiWorkflowAction - AI operations + API resource management
4. CLI - User interface layer
"""

# New layered architecture exports
from .dsl_file import DifyWorkflowDslFile
from .context_builder import DifyWorkflowContextBuilder
from .ai_workflow_action import AiWorkflowAction

# Shared models
from .models import (
    NodeConnection, NodeInfo, WorkflowInfo, ValidationError,
    NodeValidationResult, WorkflowValidationResult, LinearityCheck,
    WorkflowContext, GenerationResult, DSLValidationReport,
    DSLValidationSummary, CLICommand, NodeRecommendation,
    RecommendationResult
)

__version__ = "1.0.0"

# Clean architecture exports
__all__ = [
    # Core layered components
    'DifyWorkflowDslFile',
    'DifyWorkflowContextBuilder', 
    'AiWorkflowAction',
    
    # Shared models
    'NodeConnection', 'NodeInfo', 'WorkflowInfo', 'ValidationError',
    'NodeValidationResult', 'WorkflowValidationResult', 'LinearityCheck',
    'WorkflowContext', 'GenerationResult', 'DSLValidationReport',
    'DSLValidationSummary', 'CLICommand', 'NodeRecommendation',
    'RecommendationResult'
]