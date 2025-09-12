## 类型检查

```bash
ty check --exclude resources
```

保证每次编辑后都能通过类型检查，运行上面的命令。

## 依赖管理

```bash
uv run <script>
```

```bash
uv sync
```

```bash
uv add <package>
```

尽量使用`uv`，不要使用`pip`或者`python`。

## 类型安全

使用严格的类型标记。

只要有可能就不要使用`Any`和`Dict`。

尽可能使用pydantic model。

## 架构设计

项目采用分层架构设计，从底层到顶层分为四层：

### 第一层：DifyWorkflowDSL (dsl_model)
- 最底层的数据结构层
- 定义了Dify工作流的完整pydantic模型
- 位置：`src/dsl_model/`
- 职责：封装底层数据结构和验证逻辑

### 第二层：DifyWorkflowDslFile
- 工作流文件操作层
- 封装DifyWorkflowDSL，提供基本操作（加载、保存、修改、验证）
- 实现RAII模式进行资源管理
- 位置：`src/ai_workflow_action/dsl_file.py`
- 职责：文件I/O、基本工作流操作、数据验证

### 第三层：AiWorkflowAction
- AI操作层
- 通过DifyWorkflowContextBuilder提供AI支持的工作流操作
- 管理Anthropic API资源
- 位置：`src/ai_workflow_action/ai_workflow_action.py`
- 职责：AI节点生成、智能推荐、工作流分析

### 第四层：CLI
- 用户界面层
- 使用AiWorkflowAction提供命令行接口
- 位置：`cli.py`
- 职责：用户交互、命令解析、结果展示

### 辅助组件：DifyWorkflowContextBuilder
- 专门的上下文构建器
- 为AI操作提供高质量的提示和上下文
- 位置：`src/ai_workflow_action/context_builder.py`
- 职责：工作流分析、上下文提取、AI提示生成

### 使用模式

```python
# 推荐的使用方式（新架构）
from ai_workflow_action import AiWorkflowAction, DifyWorkflowDslFile

# 创建工作流文件对象（RAII模式）
with DifyWorkflowDslFile("workflow.yml") as dsl_file:
    # 创建AI操作对象
    with AiWorkflowAction(dsl_file) as ai_action:
        # 执行AI操作
        node_id = ai_action.auto_generate_next_node()
        # 保存更改
        ai_action.save_workflow()

# 或者简化版本
ai_action = AiWorkflowAction()
ai_action.load_workflow("workflow.yml")
ai_action.auto_generate_next_node()
ai_action.save_workflow()
```

### 简化的API设计

新架构提供了简洁统一的API接口：

```python
# 基本文件操作
dsl_file = DifyWorkflowDslFile("workflow.yml")
info = dsl_file.get_workflow_info()
validation = dsl_file.validate_workflow()

# AI增强操作  
ai_action = AiWorkflowAction(dsl_file)
analysis = ai_action.analyze_workflow()
new_node_id = ai_action.auto_generate_next_node()
suggestions = ai_action.suggest_improvements()
```