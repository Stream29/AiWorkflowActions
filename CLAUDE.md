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

严禁在任何地方使用`has_attr`和`getattr`。

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
- 基于 `cmd.Cmd` + `argparse` 架构，支持复杂命令参数解析
- 位置：`cli.py`
- 职责：用户交互、命令解析、结果展示
- 特性：
  - 自动补全和历史记录
  - 内置帮助系统
  - 优雅的错误处理
  - 支持复杂参数（选项、标志、位置参数等）
  - 自动生成使用说明

### 辅助组件：DifyWorkflowContextBuilder
- 专门的上下文构建器
- 为AI操作提供高质量的提示和上下文
- 位置：`src/ai_workflow_action/context_builder.py`
- 职责：工作流分析、上下文提取、AI提示生成

## CLI命令

### 基本命令
```bash
# 启动交互式模式
uv run python cli.py

# 直接加载文件
uv run python cli.py resources/SimpleDsl.yml

# 验证所有资源文件
uv run python cli.py --validate-resources
```

### 交互式命令

```
load <file_path>                          - 加载和验证工作流文件
save <file_path>                          - 保存工作流到文件
nodes [--verbose]                         - 列出工作流中的所有节点
  --verbose, -v                           - 显示详细连接信息
generate --after <node_id> --type <type> [--title <title>]  - 使用AI生成新节点
  --after <node_id>                       - 在指定节点后添加新节点
  --type <node_type>                      - 节点类型 (如: llm, code, http-request)
  --title <title>                         - 可选的自定义节点标题
validate_resources [--dir <directory>]   - 验证DSL文件
  --dir <directory>                       - 自定义验证目录
help [command]                            - 显示帮助信息
quit/exit                                 - 退出程序
```

## 支持的节点类型

支持以下节点类型：

- `start` - 开始节点
- `end` - 结束节点
- `answer` - 回答节点
- `llm` - LLM节点
- `code` - 代码执行节点
- `http-request` - HTTP请求节点
- `tool` - 工具调用节点
- `if-else` - 条件分支节点
- `template-transform` - 模板转换节点
- `variable-assigner` - 变量赋值节点
- `knowledge-retrieval` - 知识检索节点
- `agent` - 智能体节点
- `iteration` - 迭代节点
- `parameter-extractor` - 参数提取节点
- `question-classifier` - 问题分类节点
- `iteration-start` - 迭代开始节点
- `loop-start` - 循环开始节点
- `loop-end` - 循环结束节点
- `variable-aggregator` - 变量聚合节点
- `document-extractor` - 文档提取节点
- `list-operator` - 列表操作节点
- `` - 注释节点，记作空

## 环境变量

项目需要配置以下环境变量：

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

此变量在项目根目录的 `.env` 文件中。
