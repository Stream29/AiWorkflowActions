# AiWorkflowActions 项目信息（CLAUDE 指南）

最后更新：2025-09-12 16:40

## 项目简介
AiWorkflowActions 是一个面向 Dify 工作流（Workflow DSL）的工具集：
- 提供基于 Pydantic v2 的严格/兼容并重的 DSL 解析与校验模型（位于 `src/dsl_model`）。
- 提供命令行 CLI，支持加载/保存/校验工作流，以及对资源库中的示例 DSL 进行批量校验并生成报告。
- 聚焦于“简化使用 + 更强校验 + 易于扩展”的开发体验。

## 关键特性
- 新的 DSL 模型：使用 `dsl_model` 模块（取代历史的 `dify_core` 解析路径）。
- 图结构校验：在 `src/dsl_model/graph.py` 中对工作流图进行结构性检查（例如唯一的起始节点等）。
- 资源校验报告：可对 `resources/Awesome-Dify-Workflow/DSL` 下的全部示例进行校验，自动生成 Markdown 报告。
- 直观 CLI：无侵入、纯 UI 层封装，调度核心模块完成实际能力。

## 运行环境与依赖
- Python: `>=3.13`（见 `pyproject.toml`）
- 依赖管理：使用 `uv`（更快的 Python 包管理器）。
- 类型检查：使用 `ty`。

### 使用 uv 管理依赖
- 安装依赖：
  - `uv sync`
- 运行命令：
  - `uv run python cli.py --help`

如不使用 uv，也可在已满足依赖的环境中直接运行：
- `python cli.py --help`

## CLI 用法速览
常见命令均在 `cli.py` 中实现，入口类位于 `src/ai_workflow_action`。

- 加载并查看工作流信息：
  - `python cli.py resources/Awesome-Dify-Workflow/DSL/AgentFlow.yml`
- 校验当前已加载的工作流：
  - `python cli.py resources/.../YourDsl.yml --validate`
- 批量校验资源库并生成报告：
  - `python cli.py --validate-resources`
  - 成功与失败均会在控制台输出摘要；若存在失败，将在项目根目录生成 `DIFY_DSL_VALIDATION_REPORT.md`。
- 交互式模式：
  - `python cli.py` 后输入 `help` 查看可用命令（如：load/save/validate/nodes/auto-next 等）。

## 与 Dify DSL 相关的文件
- 新版 DSL 总入口模型：`src/dsl_model/dsl.py`（`DifyWorkflowDSL`）。
- 图与节点定义：`src/dsl_model/graph.py`、`src/dsl_model/nodes.py`、`src/dsl_model/core.py`、`src/dsl_model/enums.py`、`src/dsl_model/features.py`。
- 校验器（组合使用模型进行校验）：`src/ai_workflow_action/validator.py`。
- CLI：`cli.py`。
- 示例 DSL：`resources/Awesome-Dify-Workflow/DSL`（含若干子目录）。

## 校验报告与调查文档
- 最近一次批量校验报告：`DIFY_DSL_VALIDATION_REPORT.md`
- 早前的错误根因调查：`DIFY_DSL_VALIDATION_INVESTIGATION.md`
  - 结论：我们已将节点数据改为使用 Pydantic 的 discriminator（基于 `data.type`）进行判别，修复了“起始节点被误判”的问题。

## 项目结构（节选）
- `src/ai_workflow_action`：工作流读写、校验、上下文构建、节点生成等核心逻辑（非 LLM 能力本体）。
- `src/dsl_model`：Pydantic v2 的 DSL 数据模型（当前权威解析/校验来源）。
- `src/dify_core`：历史保留代码，不再作为默认解析路径。
- `resources/Awesome-Dify-Workflow/DSL`：示例 DSL 集合。

## 开发提示
- 若需扩展某类节点的数据结构，请在 `src/dsl_model/nodes.py` 中新增/调整对应的 `...NodeData` 并确保其包含 `type: Literal[...]`，以便经由 discriminator 正确分发。
- 修改图结构规则时，请同步更新 `src/dsl_model/graph.py` 中的模型校验逻辑，并通过 `--validate-resources` 批量验证。
- 推荐在提交前运行：
  - `uv run python cli.py --validate-resources`

## 版权与致谢
- 本项目由 JetBrains Autonomous Programmer（代号 Junie）在自动化环境下维护与演进。
- 感谢 Dify 社区生态提供的开放 DSL 示例与灵感。 