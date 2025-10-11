"""
Pydantic models for workflow operations and CLI responses.
Strong typing to replace Dict[str, Any] usage throughout the codebase.
"""

from typing import Optional, List, Dict

from pydantic import BaseModel, Field

from dsl_model import NodeData


class NodeConnection(BaseModel):
    """Node connections (incoming/outgoing)"""
    incoming: List[str] = Field(default_factory=list, description="List of incoming node IDs")
    outgoing: List[str] = Field(default_factory=list, description="List of outgoing node IDs")


class NodeInfo(BaseModel):
    """Node information for display and operations"""
    id: str = Field(description="Unique node identifier")
    title: str = Field(description="Node display title")
    type: str = Field(description="Node type")
    data: NodeData = Field(description="Node data payload")
    successor_nodes: List[str] = Field(default_factory=list, description="Successor node IDs (outgoing edges)")
    predecessor_nodes: List[str] = Field(default_factory=list, description="Predecessor node IDs (incoming edges)")
    connections: Optional[NodeConnection] = Field(default=None,
                                                  description="Node connections (deprecated - use successor/predecessor)")


class WorkflowInfo(BaseModel):
    """Workflow summary information"""
    app_name: str = Field(description="Application name")
    description: str = Field(description="Workflow description")
    mode: str = Field(description="Workflow mode (workflow, chat, etc.)")
    node_count: int = Field(ge=0, description="Total number of nodes")
    edge_count: int = Field(ge=0, description="Total number of edges")
    node_types: Dict[str, int] = Field(default_factory=dict, description="Node type counts")


class NodeValidationResult(BaseModel):
    """Node validation result"""
    node_id: str = Field(description="Node identifier")
    is_valid: bool = Field(description="Whether node is valid")
    errors: List[str] = Field(default_factory=list, description="Validation error messages")


class WorkflowValidationResult(BaseModel):
    """Complete workflow validation result"""
    is_valid: bool = Field(description="Overall workflow validity")
    structure_errors: List[str] = Field(default_factory=list, description="Structural errors")
    node_errors: Dict[str, List[str]] = Field(default_factory=dict, description="Node-specific errors")
    graph_errors: List[str] = Field(default_factory=list, description="Graph validation errors")


class LinearityCheck(BaseModel):
    """Result of linearity validation"""
    is_linear: bool = Field(description="Whether workflow is linear")
    error_message: Optional[str] = Field(default=None, description="Error message if not linear")


class WorkflowContext(BaseModel):
    """Context for AI generation"""
    app_name: str = Field(description="Application name")
    description: str = Field(description="Workflow description")
    mode: str = Field(description="Workflow mode")
    node_sequence: List[NodeInfo] = Field(description="Sequence of nodes for context")


class GenerationResult(BaseModel):
    """Result of AI node generation"""
    success: bool = Field(description="Whether generation succeeded")
    node_id: Optional[str] = Field(default=None, description="Generated node ID if successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    attempts: int = Field(default=1, description="Number of generation attempts")
    validation_errors: List[str] = Field(default_factory=list, description="Final validation errors")


class DSLValidationReport(BaseModel):
    """DSL validation report entry"""
    file: str = Field(description="Relative file path")
    errors: List[str] = Field(description="Validation errors for this file")


class DSLValidationSummary(BaseModel):
    """Summary of DSL validation across multiple files"""
    total_files: int = Field(ge=0, description="Total files scanned")
    passed: int = Field(ge=0, description="Files that passed validation")
    failed: int = Field(ge=0, description="Files that failed validation")
    failures: List[DSLValidationReport] = Field(default_factory=list, description="Detailed failure reports")


class CLICommand(BaseModel):
    """Base CLI command structure"""
    command: str = Field(description="Command name")
    args: List[str] = Field(default_factory=list, description="Command arguments")


class NodeRecommendation(BaseModel):
    """Node type recommendation"""
    node_type: str = Field(description="Recommended node type")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Reason for recommendation")


class RecommendationResult(BaseModel):
    """Result of node type recommendation"""
    recommendations: List[NodeRecommendation] = Field(description="List of recommended node types")
    selected_type: Optional[str] = Field(default=None, description="Auto-selected node type")


class TopologyNodeData(BaseModel):
    """Node data with topology information for context building"""
    id: str = Field(description="Node identifier")
    data: NodeData = Field(description="Node data payload")
    successor_nodes: List[str] = Field(default_factory=list, description="Successor node IDs")
    predecessor_nodes: List[str] = Field(default_factory=list, description="Predecessor node IDs")


class NodeOutputInfo(BaseModel):
    """Node information with outputs for prompt generation"""
    id: str = Field(description="Node identifier")
    type: str = Field(description="Node type")
    title: str = Field(description="Node title")
    successors: List[str] = Field(default_factory=list, description="Successor node IDs")
    predecessors: List[str] = Field(default_factory=list, description="Predecessor node IDs")
    outputs: Optional[List[str]] = Field(default=None, description="Available output variables")
