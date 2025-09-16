from typing import Type, Mapping

from dsl_model import NodeType, BaseNodeData
from dsl_model.nodes import (
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
    IterationStartNodeData,
    LoopStartNodeData,
    LoopEndNodeData,
    VariableAggregatorNodeData,
    ParameterExtractorNodeData,
    QuestionClassifierNodeData,
    DocumentExtractorNodeData,
    ListOperatorNodeData,
    NoteNodeData,
)

_node_enum_to_models: Mapping[NodeType, Type[BaseNodeData]] = {
    NodeType.START: StartNodeData,
    NodeType.END: EndNodeData,
    NodeType.ANSWER: AnswerNodeData,
    NodeType.LLM: LLMNodeData,
    NodeType.CODE: CodeNodeData,
    NodeType.HTTP_REQUEST: HTTPRequestNodeData,
    NodeType.TOOL: ToolNodeData,
    NodeType.IF_ELSE: IfElseNodeData,
    NodeType.TEMPLATE_TRANSFORM: TemplateTransformNodeData,
    NodeType.VARIABLE_ASSIGNER: VariableAssignerNodeData,
    NodeType.LEGACY_VARIABLE_AGGREGATOR: VariableAssignerNodeData,  # Backward compatibility
    NodeType.KNOWLEDGE_RETRIEVAL: KnowledgeRetrievalNodeData,
    NodeType.AGENT: AgentNodeData,
    NodeType.ITERATION: IterationNodeData,
    NodeType.ITERATION_START: IterationStartNodeData,
    NodeType.LOOP_START: LoopStartNodeData,
    NodeType.LOOP_END: LoopEndNodeData,
    NodeType.VARIABLE_AGGREGATOR: VariableAggregatorNodeData,
    NodeType.PARAMETER_EXTRACTOR: ParameterExtractorNodeData,
    NodeType.QUESTION_CLASSIFIER: QuestionClassifierNodeData,
    NodeType.DOCUMENT_EXTRACTOR: DocumentExtractorNodeData,
    NodeType.LIST_OPERATOR: ListOperatorNodeData,
    NodeType.NOTE: NoteNodeData,
}


class NodeTypeUtil:
    @staticmethod
    def get_node_data_model(node_type: NodeType) -> Type[BaseNodeData]:
        return _node_enum_to_models[node_type]
