from typing import Any, Optional
from pydantic import BaseModel, Field


class GraphEdge(BaseModel):
    """Graph edge entity"""
    source_node_id: str
    target_node_id: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None


class GraphParallel(BaseModel):
    """Graph parallel entity"""
    id: str
    start_node_id: str
    end_node_id: str


class AnswerStreamGenerateRoute(BaseModel):
    """Answer stream generate route"""
    answer_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    answer_generate_route: dict[str, list[str]] = Field(default_factory=dict)


class EndStreamParam(BaseModel):
    """End stream param"""
    end_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    end_stream_variable_selector_mapping: dict[str, list[list[str]]] = Field(default_factory=dict)


class Graph(BaseModel):
    root_node_id: str = Field(..., description="root node id of the graph")
    node_ids: list[str] = Field(default_factory=list, description="graph node ids")
    node_id_config_mapping: dict[str, dict] = Field(default_factory=dict)
    edge_mapping: dict[str, list[GraphEdge]] = Field(default_factory=dict)
    reverse_edge_mapping: dict[str, list[GraphEdge]] = Field(default_factory=dict)
    parallel_mapping: dict[str, GraphParallel] = Field(default_factory=dict)
    node_parallel_mapping: dict[str, str] = Field(default_factory=dict)
    answer_stream_generate_routes: AnswerStreamGenerateRoute = Field(default_factory=AnswerStreamGenerateRoute)
    end_stream_param: EndStreamParam = Field(default_factory=EndStreamParam)

    @classmethod
    def init(cls, graph_config: dict[str, Any]) -> "Graph":
        """Initialize graph from config"""
        # Simplified initialization - just find root node
        nodes = graph_config.get("nodes", [])
        edges = graph_config.get("edges", [])
        
        # Find start node as root
        root_node_id = None
        for node in nodes:
            if node.get("data", {}).get("type") == "start":
                root_node_id = node.get("id")
                break
        
        if not root_node_id:
            raise ValueError("No start node found in graph")
        
        # Build basic mappings
        node_ids = [node.get("id") for node in nodes if node.get("id")]
        node_id_config_mapping = {node.get("id"): node for node in nodes if node.get("id")}
        
        # Build edge mappings
        edge_mapping = {}
        reverse_edge_mapping = {}
        
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                graph_edge = GraphEdge(
                    source_node_id=source,
                    target_node_id=target,
                    source_handle=edge.get("sourceHandle"),
                    target_handle=edge.get("targetHandle")
                )
                
                if source not in edge_mapping:
                    edge_mapping[source] = []
                edge_mapping[source].append(graph_edge)
                
                if target not in reverse_edge_mapping:
                    reverse_edge_mapping[target] = []
                reverse_edge_mapping[target].append(graph_edge)
        
        return cls(
            root_node_id=root_node_id,
            node_ids=node_ids,
            node_id_config_mapping=node_id_config_mapping,
            edge_mapping=edge_mapping,
            reverse_edge_mapping=reverse_edge_mapping
        )